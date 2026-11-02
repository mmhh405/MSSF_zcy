"""Inference-only attention-gate intervention on one fixed trained V2 checkpoint."""
import argparse
import copy
import fcntl
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from mmengine import load
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmdet3d.engine.hooks.reliability_experiment_hook import _atomic_write
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules
from reliability_robustness import (ROOT, SETTINGS, GateObserver,
                                    align_sample_indices, corrupt_batch,
                                    file_hash, uniform_indices)


POLICIES = {
    'learned': (),
    'middle_identity': ('middle_encoder',),
    'backbone_identity': ('backbone',),
    'both_identity': ('middle_encoder', 'backbone'),
}
CONDITIONS = ('clean', 'dark', 'blur', 'occlusion')
START = '<!-- ATTENTION_INTERVENTION_START -->'
END = '<!-- ATTENTION_INTERVENTION_END -->'


def apply_intervention(model, policy):
    """Change only selected final gate logits; keep attention and query gates."""
    gates = {name: module for name, module in model.named_modules()
             if module.__class__.__name__ == 'ReliabilityGateV2'}
    attention = {name: gate for name, gate in gates.items()
                 if 'query_reliability_gate' not in name}
    if len(gates) != 4 or len(attention) != 2:
        raise ValueError('Expected a full V2 model with two query/two attention gates')
    for stage in ('middle_encoder', 'backbone'):
        if sum(name.startswith(stage + '.') for name in attention) != 1:
            raise ValueError(f'Expected exactly one attention gate in {stage}')
    before = {name: {key: value.detach().clone()
                     for key, value in gate.state_dict().items()}
              for name, gate in gates.items()}
    selected = [name for name in attention
                if name.split('.')[0] in POLICIES[policy]]
    allowed, changed = set(), set()
    with torch.no_grad():
        for name in selected:
            gate = gates[name]
            last = len(gate.predictor) - 1
            gate.predictor[last].weight.zero_()
            gate.predictor[last].bias.zero_()
            allowed.update((name, f'predictor.{last}.{key}')
                           for key in ('weight', 'bias'))
    for name, gate in gates.items():
        for key, value in gate.state_dict().items():
            if not torch.equal(value, before[name][key]):
                changed.add((name, key))
    if not changed.issubset(allowed):
        raise AssertionError('Intervention changed a parameter outside selected logits')
    return dict(identity_gate_names=selected,
                changed_parameter_keys=sorted(f'{name}.{key}' for name, key in changed),
                query_parameters_unchanged=True, mutation_scope_checked=True,
                semantics='selected valid gates=2*sigmoid(0)=1; invalid mask retained; '
                          'attention branch retained; checkpoint is never rewritten')


def save_report(path, report):
    _atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def record_full_result(report):
    archive = ROOT / 'repro/reliability_fusion_v2/attention_intervention' / (report['run_id'] + '.json')
    target = ROOT / 'README_reliability_fusion_v2.md'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(archive.parent / '.record.lock')):
        rows = ['| 条件 | 策略 | EAA (%) | 相对 learned | DC (%) | 相对 learned |',
                '|---|---|---:|---:|---:|---:|']
        for condition in report['conditions']:
            baseline = next(row for row in report['results']
                            if row['policy'] == 'learned' and row['condition'] == condition)
            for row in report['results']:
                if row['condition'] == condition:
                    rows.append(f"| {condition} | {row['policy']} | {row['eaa_map']:.4f} | "
                                f"{row['eaa_map']-baseline['eaa_map']:+.4f} | "
                                f"{row['dc_map']:.4f} | {row['dc_map']-baseline['dc_map']:+.4f} |")
        block = ('\n\n这是同一个完整 V2 checkpoint 的推理干预，无训练；'
                 '两个 query 门控保持学习值，选中的注意力输出门控仅在内存中设为恒等。'
                 '不能当作重新训练后的结构消融。每组同一 1296 帧，普通 3D AP。\n\n'
                 + '\n'.join(rows) + '\n\n[逐类指标、门控分布及不可变权重核验]('
                 + archive.relative_to(ROOT).as_posix() + ')\n\n')
        text = target.read_text(encoding='utf-8')
        if START not in text:
            text += '\n## 固定权重注意力门控干预（自动更新）\n\n' + START + END + '\n'
        prefix, remainder = text.split(START)
        _, suffix = remainder.split(END)
        save_report(archive, report)
        _atomic_write(target, prefix + START + block + END + suffix)


def evaluate(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the project root')
    if not torch.cuda.is_available():
        raise RuntimeError('The configured CUDA environment is required')
    output = Path(args.work_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('progress.json', 'summary.json')):
        raise FileExistsError('Use a new work directory; existing results are preserved')
    conditions = ['clean'] + [c for c in dict.fromkeys(args.conditions) if c != 'clean']
    annotations = load('data/vod/kitti_infos_val.pkl')['data_list']
    indices = uniform_indices(len(annotations), args.samples)
    full = len(indices) == len(annotations)
    if args.record_readme and (not full or set(conditions) != set(CONDITIONS)):
        raise ValueError('Only full-validation four-condition runs may update the README')
    register_all_modules()
    MMLogger.get_instance('attention_intervention', log_file=str(output / 'metrics.log'))
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.indices = indices
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.dump(str(output / 'resolved.py'))
    report = dict(
        run_id=datetime.now().strftime('%Y%m%d_%H%M%S'), valid=False,
        experiment_type='fixed-checkpoint inference intervention', training=False,
        full_validation=full, samples=len(indices), indices=indices, conditions=conditions,
        policies=list(POLICIES), seed=0, corruption_settings=SETTINGS,
        original_frame_alignment_checked=True, query_gates='learned and unchanged',
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_hash, strict_load=True),
        config=dict(path=args.config, sha256=file_hash(args.config)),
        annotation_sha256=file_hash('data/vod/kitti_infos_val.pkl'),
        script_sha256=file_hash(__file__),
        helper_sha256=file_hash(Path(__file__).with_name('reliability_robustness.py')),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty_at_start=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        interventions={}, results=[], output_dir=str(output))
    save_report(output / 'progress.json', report)
    for policy in POLICIES:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        model = MODELS.build(cfg.model).cuda().eval()
        model.load_state_dict(state, strict=True)
        report['interventions'][policy] = apply_intervention(model, policy)
        loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
        for condition in conditions:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            directory = output / policy / condition
            directory.mkdir(parents=True, exist_ok=True)
            metrics = copy.deepcopy(cfg.test_evaluator)
            for metric in metrics:
                metric['pklfile_prefix'] = str(directory / 'predictions')
            evaluator = Evaluator(metrics)
            evaluator.dataset_meta = loader.dataset.metainfo
            observer = GateObserver(model)
            offset = 0
            try:
                with torch.no_grad():
                    for batch_index, data in enumerate(loader):
                        data = corrupt_batch(data, condition)
                        predictions = model.test_step(data)
                        offset = align_sample_indices(predictions, indices, offset, annotations)
                        evaluator.process(data_samples=predictions, data_batch=data)
                        if (batch_index + 1) % 64 == 0:
                            print(f'{policy}/{condition}: {offset}/{len(indices)}', flush=True)
            finally:
                gate_stats = observer.finish()
            if offset != len(indices):
                raise AssertionError('Prediction count differs from selected annotations')
            for name in report['interventions'][policy]['identity_gate_names']:
                if gate_stats[name]['valid'] and not (
                        gate_stats[name]['minimum'] == gate_stats[name]['maximum'] == 1.):
                    raise AssertionError(f'Valid identity gate is not 1: {name}')
            scores = {key: float(value) for key, value in evaluator.evaluate(len(indices)).items()}
            row = dict(policy=policy, condition=condition, metrics=scores, gates=gate_stats,
                       eaa_map=next(v for k, v in scores.items() if k.endswith('entire_area/map')),
                       dc_map=next(v for k, v in scores.items() if k.endswith('roi/map')))
            report['results'].append(row)
            save_report(output / 'progress.json', report)
            print('RESULT', policy, condition, row['eaa_map'], row['dc_map'], flush=True)
        del model, loader
        torch.cuda.empty_cache()
    report['checkpoint_sha256_after'] = file_hash(checkpoint)
    if report['checkpoint_sha256_after'] != checkpoint_hash:
        raise AssertionError('Source checkpoint changed during evaluation')
    report['checkpoint_unchanged'] = True
    report['completed_at'] = datetime.now().astimezone().isoformat()
    report['valid'] = True
    save_report(output / 'summary.json', report)
    save_report(output / 'progress.json', report)
    if args.record_readme:
        record_full_result(report)
    print('COMPLETED', output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/reliability_fusion_v2/gate_main.py')
    parser.add_argument('--checkpoint', default='work_dirs/reliability_fusion_v2/gate_main/20261030_214524/epoch_6.pth')
    parser.add_argument('--samples', type=int, default=1296)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--record-readme', action='store_true')
    args = parser.parse_args()
    lock_path = ROOT / 'work_dirs/reliability_fusion_v2/.gpu0-queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('GPU queue is already locked; wait for the active experiment') from error
        evaluate(args)


if __name__ == '__main__':
    main()
