"""Fixed-checkpoint VoD evaluations for the V3 closeout."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from mmengine import load
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmdet3d.evaluation.functional.kitti_utils.eval_vod import (
    calculate_iou_partly, clean_data, compute_statistics_jit, eval_class,
    get_m_ap, get_m_ap_r40)
from mmdet3d.evaluation.metrics.vod_metric import VODMetric
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'tools/analysis_tools'))
import reliability_robustness as robust

ANN = ROOT / 'data/vod/kitti_infos_val.pkl'
MASK = (.175, .175, .825, .825)
IOU = 0.25


def save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def mask_image(image, condition):
    assert condition == 'center_42' and image.ndim == 3 and image.shape[0] == 3
    height, width = image.shape[-2:]
    x0, y0, x1, y1 = MASK
    masked = image.clone()
    masked[:, round(y0 * height):round(y1 * height),
           round(x0 * width):round(x1 * width)] = 0
    return masked


def table(annotation, is_detection=False):
    columns = [np.asarray(annotation['bbox']).reshape(-1, 4), annotation['alpha']]
    if is_detection:
        columns.append(annotation['score'])
    return np.column_stack(columns).astype('float64')


def operating_points(detections, ground_truth, class_id, roi, thresholds):
    overlaps = calculate_iou_partly(detections, ground_truth, 2,
                                    num_parts=min(130, len(detections)))[0]
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for overlap, gt, dt in zip(overlaps, ground_truth, detections):
        _, ignored_gt, ignored_dt, dont_care = clean_data(gt, dt, class_id, 0, roi_clean=roi)
        gt_data, dt_data = table(gt), table(dt, is_detection=True)
        ignored_gt = np.asarray(ignored_gt, dtype='int64')
        ignored_dt = np.asarray(ignored_dt, dtype='int64')
        dont_care = np.asarray(dont_care, dtype='float64').reshape(-1, 4)
        for i, threshold in enumerate(thresholds):
            counts[i] += compute_statistics_jit(
                overlap, gt_data, dt_data, ignored_gt, ignored_dt, dont_care,
                2, IOU, threshold, True, False)[:3]
    result = []
    for threshold, (tp, fp, fn) in zip(thresholds, counts.tolist()):
        result.append(dict(score_threshold=threshold, tp=tp, fp=fp, fn=fn,
                           precision_pct=100 * tp / max(1, tp + fp),
                           recall_pct=100 * tp / max(1, tp + fn),
                           fp_per_frame=fp / len(detections)))
    return result


def cyclist_curve(detections, gt):
    values = eval_class(gt, detections, [2], [0], 2,
                        np.full((1, 3, 1), IOU), custom_method=3,
                        num_parts=min(50, len(gt)))
    precision = values['precision'][0, 0, 0]
    recall = values['recall'][0, 0, 0]
    return dict(ap11=float(get_m_ap(precision)), ap40=float(get_m_ap_r40(precision)),
                precision=precision.tolist(), recall=recall.tolist(),
                nominal_recall_grid=np.linspace(0, 1, len(precision)).tolist(),
                ap11_slot_indices=list(range(0, len(precision), 4)),
                operating_points=operating_points(detections, gt, 2, True,
                                                  [.1, .13, .2, .3, .4, .5]))


def evaluations():
    folder = ROOT / 'work_dirs/reliability_fusion_v3'
    dense = Path((folder / 'last_dense_control_repeats_path.txt').read_text().strip())
    dense_seed0 = Path((folder / 'last_paired_path.txt').read_text().strip()) / 'dense_control'
    repeats = Path((folder / 'last_prior_repeats_path.txt').read_text().strip())
    assert 'ALL_COMPLETED' in (dense / 'queue.log').read_text()
    assert 'ALL_COMPLETED' in (repeats / 'queue.log').read_text()
    result = {}
    for pair in json.loads((repeats / 'comparison.json').read_text())['pairs']:
        seed = pair['seed']
        result[seed] = {
            'dense': dense_seed0 if seed == 0 else dense / f'seed{seed}',
            'guided': ROOT / pair['arms']['guided']['work_dir'],
            'independent': ROOT / pair['arms']['independent']['work_dir'],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_ann = load(ANN)
    annotations = raw_ann['data_list']
    assert len(annotations) == 1296
    indices = robust.uniform_indices(len(annotations), 32 if args.smoke else len(annotations))
    settings = dict(annotation_sha256=robust.file_hash(ANN), frames=len(indices),
                    indices=indices, image_mask_xyxy=MASK, pedestrian_score=.13,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (output / 'protocol.json').is_file():
        if json.loads((output / 'protocol.json').read_text()) != json.loads(json.dumps(settings)):
            raise RuntimeError('Evaluation protocol has changed')
    else:
        save(output / 'protocol.json', settings)
    register_all_modules()
    MMLogger.get_instance('v3_closeout', log_file=str(output / 'evaluator.log'))
    robust.corrupt_image = mask_image
    torch.set_num_threads(4)
    runs = evaluations()
    jobs = [(2, 'dense', 5, ['clean']), (2, 'dense', 6, ['clean', 'center_42']),
            (2, 'guided', 6, ['clean', 'center_42'])]
    jobs.extend((seed, arm, 6, ['center_42'])
                for seed in (0, 1) for arm in ('dense', 'guided'))
    if args.smoke:
        jobs = [jobs[1], jobs[2]]

    results = []
    for seed, arm, epoch, conditions in jobs:
        run = runs[seed][arm]
        checkpoint = run / f'epoch_{epoch}.pth'
        checkpoint_hash = robust.file_hash(checkpoint)
        config_path, = run.glob('*.py')
        if all((output / f'seed{seed}' / arm / f'epoch{epoch}' / c / 'metrics.json').is_file()
               for c in conditions):
            model = None
        else:
            cfg = Config.fromfile(str(config_path))
            import_modules_from_strings(**cfg.custom_imports)
            assert cfg.randomness.seed == seed
            assert cfg.model.test_cfg.nms_pre == 300 and cfg.model.test_cfg.score_thr == .1
            assert cfg.model.early_budget.mode == ('dense' if arm == 'dense' else 'select')
            cfg.test_dataloader.dataset.indices = indices
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False
            model = MODELS.build(cfg.model)
            weights = torch.load(checkpoint, map_location='cpu')
            assert weights['meta']['epoch'] == epoch and weights['meta']['iter'] == 5139 * epoch
            model.load_state_dict(weights['state_dict'], strict=True)
            del weights
            model.cuda().eval()
            loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
            all_gt = VODMetric(ann_file=str(ANN)).convert_annos_to_kitti_annos(raw_ann)
            gt = [all_gt[index]['kitti_annos'] for index in indices]

        for condition in conditions:
            directory = output / f'seed{seed}' / arm / f'epoch{epoch}' / condition
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'metrics.json'
            if path.is_file():
                previous = json.loads(path.read_text())
                assert previous['checkpoint_sha256'] == checkpoint_hash
                results.append(previous)
                print('REUSE', seed, arm, epoch, condition, flush=True)
                continue
            print('START', seed, arm, epoch, condition, flush=True)
            start = time.monotonic()
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            cfg.dump(str(directory / 'resolved.py'))
            metric = copy.deepcopy(cfg.test_evaluator)
            metric[0]['pklfile_prefix'] = str(directory / 'predictions')
            evaluator = Evaluator(metric)
            evaluator.dataset_meta = loader.dataset.metainfo
            offset = 0
            with torch.no_grad():
                for data in loader:
                    if condition != 'clean':
                        data = robust.corrupt_batch(data, condition)
                    predictions = model.test_step(data)
                    offset = robust.align_sample_indices(predictions, indices, offset, annotations)
                    evaluator.process(data_samples=predictions, data_batch=data)
                    if offset % 256 == 0:
                        print('PROGRESS', seed, arm, epoch, condition, offset, len(indices), flush=True)
            assert offset == len(indices)
            metrics = {k: float(v) for k, v in evaluator.evaluate(offset).items()}
            detections = load(directory / 'predictions/pred_instances_3d.pkl')
            assert len(detections) == len(indices)
            assert all(np.all(d['sample_idx'] == i) for d, i in zip(detections, indices))
            row = dict(seed=seed, arm=arm, epoch=epoch, condition=condition,
                       checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                       seconds=round(time.monotonic() - start, 2), metrics=metrics)
            if condition == 'center_42':
                row['pedestrian_eaa'] = operating_points(detections, gt, 1, False, [.13])[0]
            else:
                row['cyclist_dc_curve'] = cyclist_curve(detections, gt)
                for key, suffix in [('ap11', '/roi/Cyclist_3d_all'),
                                    ('ap40', '/roi/Cyclist_3d_all_r40')]:
                    assert abs(row['cyclist_dc_curve'][key] -
                               next(v for k, v in metrics.items() if k.endswith(suffix))) < .00011
            save(path, row)
            results.append(row)
            print('RESULT', seed, arm, epoch, condition,
                  {k: v for k, v in metrics.items() if k.endswith('/map')}, flush=True)
        del model
        torch.cuda.empty_cache()
    save(output / 'summary.json', dict(protocol=settings, rows=results))
    print('ACCURACY_COMPLETED', output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
