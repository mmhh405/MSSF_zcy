"""Paired VoD image-corruption evaluation; checkpoints and radar stay unchanged."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from filelock import FileLock
from mmengine import load
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[2]
START = '<!-- ROBUSTNESS_RESULTS_START -->'
END = '<!-- ROBUSTNESS_RESULTS_END -->'
CONDITIONS = ('clean', 'dark', 'blur', 'occlusion', 'blackout')
SETTINGS = dict(dark_factor=.2, blur_kernel=21, blur_sigma=5.,
                occlusion='central 50% width x 50% height, black',
                blackout='all image pixels zero; not a radar-only model')


def corrupt_image(image, condition):
    """Operate on an unnormalized CHW image, preserving dtype and dimensions."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected a three-channel CHW image before preprocessing')
    if condition == 'clean':
        return image
    if condition == 'dark':
        return (image.float() * SETTINGS['dark_factor']).clamp(0, 255).to(image.dtype)
    if condition == 'blackout':
        return torch.zeros_like(image)
    if condition == 'occlusion':
        result = image.clone()
        height, width = image.shape[-2:]
        result[:, height // 4:3 * height // 4, width // 4:3 * width // 4] = 0
        return result
    array = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    array = cv2.GaussianBlur(array, (SETTINGS['blur_kernel'],) * 2,
                             SETTINGS['blur_sigma'])
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(
        device=image.device, dtype=image.dtype)


def corrupt_batch(data, condition):
    """Replace only the raw image field; keep points, labels and camera geometry."""
    if 'img' not in data['inputs']:
        raise KeyError('Expected inputs.img from Pack3DDetInputs')
    def apply(value):
        if isinstance(value, list):
            return [apply(item) for item in value]
        if isinstance(value, tuple):
            return tuple(apply(item) for item in value)
        if value.ndim == 4:
            return torch.stack([corrupt_image(item, condition) for item in value])
        return corrupt_image(value, condition)
    inputs = dict(data['inputs'])
    inputs['img'] = apply(inputs['img'])
    return dict(data, inputs=inputs)


def uniform_indices(total, count):
    if count < 1 or count > total:
        raise ValueError(f'samples must be between 1 and {total}')
    return np.rint(np.linspace(0, total - 1, count)).astype(int).tolist()


class GateObserver:
    """Read actual gate scores; repeated forward calls bypass module hooks."""
    def __init__(self, model):
        self.records = {}
        self.handles = []
        for name, module in model.named_modules():
            if module.__class__.__name__ != 'ReliabilityGate':
                continue
            self.records[name] = dict(scores=[], queries=0, valid=0)
            def observe(gate, args, output, name=name):
                _, scores = gate.forward(*args, return_gate=True)
                valid = (args[2].bool() & torch.isfinite(args[0]).all(-1, keepdim=True)
                         & torch.isfinite(args[1]).all(-1, keepdim=True)
                         & torch.isfinite(args[3]).all(-1, keepdim=True))
                record = self.records[name]
                record['queries'] += scores.numel()
                record['valid'] += int(valid.sum())
                record['scores'].append(scores[valid].detach().float().cpu())
            self.handles.append(module.register_forward_hook(observe))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        result = {}
        for name, record in self.records.items():
            scores = torch.cat(record.pop('scores'))
            count = record['queries']
            record['valid_fraction'] = record['valid'] / count if count else None
            record.update(mean=None, std=None, minimum=None, maximum=None,
                          fraction_below_0_9=None)
            if scores.numel():
                record.update(mean=float(scores.mean()), std=float(scores.std(unbiased=False)),
                              minimum=float(scores.min()), maximum=float(scores.max()),
                              fraction_below_0_9=float((scores < .9).float().mean()))
            result[name] = record
        return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_readme(report):
    """Keep training results intact; update a separate diagnostic section."""
    from mmdet3d.engine.hooks.reliability_experiment_hook import _atomic_write
    target = ROOT / 'README_reliability_fusion.md'
    archive = ROOT / 'repro/reliability_fusion/robustness' / (report['run_id'] + '.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ROOT / 'repro/reliability_fusion/robustness.lock')):
        text = target.read_text(encoding='utf-8')
        rows = ['| 条件 | 样本数 | 对照 EAA | 门控 EAA | 门控减对照 | 对照 DC | 门控 DC | 门控减对照 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition in report['conditions']:
            control, gate = (next(row for row in report['results'] if
                                 row['model'] == name and row['condition'] == condition)
                             for name in ('control', 'gate'))
            rows.append(f"| {condition} | {report['samples']} | {control['eaa_map']:.4f} | "
                        f"{gate['eaa_map']:.4f} | {gate['eaa_map']-control['eaa_map']:+.4f} | "
                        f"{control['dc_map']:.4f} | {gate['dc_map']:.4f} | "
                        f"{gate['dc_map']-control['dc_map']:+.4f} |")
        rows += ['', '| 条件 | 门控阶段 | 有效查询门控均值 | 标准差 | 低于 0.9 的比例 |',
                 '|---|---|---:|---:|---:|']
        for row in report['results']:
            if row['model'] != 'gate':
                continue
            for name, stats in row['gates'].items():
                if stats['mean'] is not None:
                    rows.append(f"| {row['condition']} | {name.split('.')[0]} | "
                                f"{stats['mean']:.7f} | {stats['std']:.7f} | "
                                f"{stats['fraction_below_0_9']:.5f} |")
        kind = '完整验证集受控退化评测' if report['full_validation'] else '子集诊断，不能与论文完整验证集分数比较'
        block = (f"\n\n{kind}。同一批 {report['samples']} 帧、固定两组第 6 epoch 权重，"
                 f"无训练。运行 `{report['run_id']}`；合成退化不等同真实夜间或恶劣天气。\n\n"
                 + '\n'.join(rows) + f"\n\n[完整参数、逐类指标与门控分布]({archive.relative_to(ROOT).as_posix()})\n\n")
        if START not in text:
            text += '\n## 受控图像退化诊断（自动更新）\n\n' + START + END + '\n'
        prefix, remainder = text.split(START)
        _, suffix = remainder.split(END)
        _atomic_write(archive, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        _atomic_write(target, prefix + START + block + END + suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                        default=['clean', 'dark', 'blur', 'occlusion'])
    parser.add_argument('--control-checkpoint', default='work_dirs/reliability_fusion/control/20261030_111629/epoch_6.pth')
    parser.add_argument('--gate-checkpoint', default='work_dirs/reliability_fusion/gate/20261030_152139/epoch_6.pth')
    parser.add_argument('--work-dir')
    parser.add_argument('--no-record-readme', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the project root')
    if not torch.cuda.is_available():
        raise RuntimeError('This VoD evaluation requires the configured CUDA environment')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(args.work_dir or f'work_dirs/reliability_fusion/robustness/{run_id}')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Use a new work directory to preserve the previous result')
    register_all_modules()
    MMLogger.get_instance('reliability_robustness', log_file=str(output / 'metrics.log'))
    conditions = ['clean'] + [c for c in dict.fromkeys(args.conditions) if c != 'clean']
    total = len(load('data/vod/kitti_infos_val.pkl')['data_list'])
    indices = uniform_indices(total, args.samples)
    report = dict(run_id=run_id, samples=len(indices), full_validation=len(indices) == total,
                  indices=indices, conditions=conditions, corruption_settings=SETTINGS,
                  seed=0, gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  annotation_sha256=file_hash('data/vod/kitti_infos_val.pkl'),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  git_dirty_at_start=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                  checkpoints={}, results=[], training=False, output_dir=str(output))
    for name, checkpoint in [('control', args.control_checkpoint), ('gate', args.gate_checkpoint)]:
        config_path = f'configs/experiments/reliability_fusion/mssf_pp_vod_{name}.py'
        cfg = Config.fromfile(config_path)
        cfg.test_dataloader.dataset.indices = indices
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
        cfg.dump(str(output / f'{name}_resolved.py'))
        report['checkpoints'][name] = dict(path=checkpoint, sha256=file_hash(checkpoint),
                                           config=config_path, config_sha256=file_hash(config_path))
        torch.manual_seed(0)
        model = MODELS.build(cfg.model).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
        loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
        for condition in conditions:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            directory = output / name / condition
            directory.mkdir(parents=True, exist_ok=True)
            metrics = copy.deepcopy(cfg.test_evaluator)
            for metric in metrics:
                metric['pklfile_prefix'] = str(directory / 'predictions')
            evaluator = Evaluator(metrics)
            evaluator.dataset_meta = loader.dataset.metainfo
            observer = GateObserver(model)
            with torch.no_grad():
                for index, data in enumerate(loader):
                    data = corrupt_batch(data, condition)
                    predictions = model.test_step(data)
                    evaluator.process(data_samples=predictions, data_batch=data)
                    if (index + 1) % 16 == 0:
                        print(f'{name}/{condition}: {index+1}/{len(indices)}', flush=True)
            scores = {key: float(value) for key, value in evaluator.evaluate(len(indices)).items()}
            row = dict(model=name, condition=condition, metrics=scores,
                       eaa_map=next(v for k, v in scores.items() if k.endswith('entire_area/map')),
                       dc_map=next(v for k, v in scores.items() if k.endswith('roi/map')),
                       gates=observer.finish())
            report['results'].append(row)
            (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            print('RESULT', name, condition, row['eaa_map'], row['dc_map'], flush=True)
        del model
        torch.cuda.empty_cache()
    for row in report['results']:
        clean = next(r for r in report['results'] if r['model'] == row['model'] and r['condition'] == 'clean')
        row['eaa_drop_from_clean'] = round(clean['eaa_map'] - row['eaa_map'], 4)
        row['dc_drop_from_clean'] = round(clean['dc_map'] - row['dc_map'], 4)
    report['completed_at'] = datetime.now().astimezone().isoformat()
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.no_record_readme:
        write_readme(report)
    print('COMPLETED', output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
