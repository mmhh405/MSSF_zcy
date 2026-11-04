"""Compare actual V3 execution and persistent FPN storage at matched inputs.

Includes data preprocessing, router, gather, feature extraction and decoding in
synchronized wall latency. Dataset loading is excluded. Pixel work is a proxy,
not FLOPs. Other GPU clients invalidate an exclusive performance comparison.
"""
import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/reliability_fusion_v3/gate_main.py')
    parser.add_argument('--checkpoint')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--allow-busy-gpu', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('samples must be positive and warmup nonnegative')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    gpu_clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                            '--format=csv,noheader'], text=True).strip()
    other_clients = [line for line in gpu_clients.splitlines()
                     if line.split(',')[0].strip() != str(os.getpid())]
    desktop_names = ('gnome-remote-desktop', 'gnome-shell', '/Xorg')
    competing_clients = [line for line in other_clients if not any(
        name in line for name in desktop_names)]
    if competing_clients and not args.allow_busy_gpu:
        raise RuntimeError('GPU has other compute clients. Run after they finish, '
                           'or use --allow-busy-gpu for a non-performance smoke check.')
    register_all_modules()
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    torch.manual_seed(cfg.get('randomness', {}).get('seed', 0))
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.dataset.indices = list(range(args.samples + args.warmup))
    loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint_path = Path(args.checkpoint or cfg.load_from)
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    weights = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not key.startswith('early_router.') for key in missing):
        raise RuntimeError(dict(missing=missing, unexpected=unexpected))
    del weights, checkpoint
    model.cuda().eval()
    summary = dict(config=args.config, checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_hash, uninitialized_router=bool(missing),
        missing_keys=missing, gpu_name=torch.cuda.get_device_name(),
        random_seed=cfg.get('randomness', {}).get('seed', 0),
        other_gpu_clients=other_clients, competing_compute_clients=competing_clients,
        performance_comparison_valid=not bool(competing_clients) and args.samples >= 32 and args.warmup >= 5,
        parameters=sum(p.numel() for p in model.parameters()),
        router_parameters=sum(p.numel() for p in model.early_router.parameters()),
        latency_scope='test_step: preprocessing through decoded predictions; dataset I/O excluded',
        pixel_work_scope='image backbone input pixels including coarse global and tile halos; excludes router',
        modes={})
    torch.set_num_threads(4)
    for mode in ('dense', 'select'):
        model.routing_mode = mode
        rows = []
        for index, data in enumerate(loader):
            batch = copy.deepcopy(data)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                predictions = model.test_step(batch)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            for prediction in predictions:
                pred = prediction.pred_instances_3d
                if not (torch.isfinite(pred.scores_3d).all() and torch.isfinite(pred.bboxes_3d.tensor).all()):
                    raise FloatingPointError('Nonfinite prediction')
            row = dict(frame_index=index, wall_ms=elapsed,
                       predictions_finite=True,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       detections=[len(p.pred_instances_3d.scores_3d) for p in predictions],
                       **model.last_budget_stats)
            if index >= args.warmup:
                rows.append(row)
            del data, batch, predictions
        summary['modes'][mode] = dict(samples=len(rows), rows=rows,
            median_wall_ms=statistics.median(row['wall_ms'] for row in rows),
            mean_wall_ms=statistics.mean(row['wall_ms'] for row in rows),
            max_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows))
        print('COMPLETED', mode, summary['modes'][mode]['median_wall_ms'], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('SAVED', output)


if __name__ == '__main__':
    main()
