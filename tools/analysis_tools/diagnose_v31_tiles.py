"""Measure selected-tile coverage of projected GT boxes after routing decisions."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmdet3d.models.layers.fusion_layers.early_budget_gate_v3 import project
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def projected_box_coverage(mask, input_shape, sample, meta, classes):
    """Use the executor's rounded tile boundaries and clipped 2D box envelopes."""
    h, w = input_shape
    rows, cols = mask.shape
    selected = mask.nonzero()
    rectangles = torch.tensor([
        [round(int(col) * w / cols) / w, round(int(row) * h / rows) / h,
         round((int(col) + 1) * w / cols) / w,
         round((int(row) + 1) * h / rows) / h]
        for row, col in selected], dtype=torch.float32).reshape(-1, 4)
    gt = sample.gt_instances_3d
    labels = gt.labels_3d.detach().cpu()
    corners = gt.bboxes_3d.corners.detach().cpu()
    centres = gt.bboxes_3d.gravity_center.detach().cpu()
    image_h, image_w = meta.get('img_shape', (h, w))[:2]
    upper = torch.tensor([image_w / w, image_h / h])
    records = []
    excluded = 0
    for label, points, centre in zip(labels.tolist(), corners, centres):
        if label < 0:
            continue
        uvz, _ = project(points, dict(meta, input_shape=(h, w)))
        if not torch.isfinite(uvz).all() or not (uvz[:, 2] > 1e-5).all():
            excluded += 1
            continue
        lo = torch.minimum(uvz[:, :2].amin(0).clamp_min(0), upper)
        hi = torch.minimum(uvz[:, :2].amax(0).clamp_min(0), upper)
        if not (hi > lo).all():
            excluded += 1
            continue
        intersection = (torch.minimum(rectangles[:, 2:], hi)
                        - torch.maximum(rectangles[:, :2], lo)).clamp_min(0)
        fraction = float(intersection.prod(-1).sum() / (hi - lo).prod())
        box_centre = (lo + hi) / 2
        centre_hit = bool(((rectangles[:, :2] <= box_centre)
                           & (rectangles[:, 2:] > box_centre)).all(-1).any())
        area = float((hi[0] - lo[0]) * w * (hi[1] - lo[1]) * h)
        distance = float(centre[:2].norm())
        records.append(dict(
            category=classes[label], distance_m=distance, projected_area_px=area,
            distance_bin='near_0_20' if distance < 20 else
                         'middle_20_40' if distance < 40 else 'far_40_plus',
            size_bin='small' if area < 32 ** 2 else
                     'medium' if area < 96 ** 2 else 'large',
            selected_area_fraction=fraction, any_tile_overlap=fraction > 0,
            half_box_covered=fraction >= .5, centre_selected=centre_hit))
    return records, excluded


def summarize(records):
    if not records:
        return dict(objects=0)
    count = len(records)
    return dict(objects=count, **{
        key: sum(row[key] for row in records) / count
        for key in ['selected_area_fraction', 'any_tile_overlap',
                    'half_box_covered', 'centre_selected']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--selection', choices=['legacy', 'index_matched', 'spatial'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    if cfg.model.early_budget.mode != 'select':
        parser.error('coverage diagnostics require a select-mode config')
    if args.selection:
        cfg.model.early_budget.router.image_selection = args.selection
    torch.manual_seed(cfg.get('randomness', {}).get('seed', 0))
    torch.set_num_threads(4)
    cfg.test_dataloader.update(batch_size=1, num_workers=0, persistent_workers=False)
    # Load GT for diagnostics while retaining validation order and empty frames.
    cfg.test_dataloader.dataset.test_mode = False
    cfg.test_dataloader.dataset.filter_empty_gt = False
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline.insert(2, dict(type='LoadAnnotations3D', with_bbox_3d=True,
                            with_label_3d=True))
    pipeline[-1]['keys'] = ['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']
    cfg.test_dataloader.dataset.indices = list(range(args.samples))
    cfg.test_dataloader.sampler.shuffle = False
    loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
    classes = loader.dataset.metainfo['classes']
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    del checkpoint
    model.cuda().eval()
    snapshots = []

    def capture(module, inputs, state):
        snapshots.append(dict(mask=state['image_mask'].detach().cpu(),
                              shape=state['input_shape'], metas=inputs[2]))

    handle = model.early_router.register_forward_hook(capture)
    frames, objects = [], []
    try:
        for data in loader:
            snapshots.clear()
            with torch.no_grad():
                predictions = model.test_step(data)
            if len(snapshots) != 1:
                raise RuntimeError('Expected one router call per frame')
            snapshot = snapshots[0]
            records, excluded = projected_box_coverage(
                snapshot['mask'][0], snapshot['shape'], data['data_samples'][0],
                snapshot['metas'][0], classes)
            stats = model.last_budget_stats
            frames.append(dict(
                sample_idx=str(predictions[0].metainfo.get('sample_idx')),
                mask=snapshot['mask'][0].int().tolist(), objects=records,
                excluded_projection_boxes=excluded,
                execution={k: stats[k] for k in [
                    'image_selected_tiles', 'image_total_tiles',
                    'image_executed_pixels', 'image_pixel_work_ratio',
                    'fpn_persistent_bytes']}))
            objects.extend(records)
    finally:
        handle.remove()
    groups = defaultdict(list)
    for row in objects:
        for field in ['category', 'distance_bin', 'size_bin']:
            groups[field + '/' + row[field]].append(row)
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    result = dict(
        config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=digest.hexdigest(),
        git_commit=__import__('subprocess').check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        selection_policy=model.early_router.image_selection,
        fixed_checkpoint_selection_override=args.selection is not None,
        description='Projected-box tile coverage, conditional on valid positive-depth '
                    'corners and visible box area. GT is read only after routing. '
                    'Runtime includes diagnostic hooks and is unsuitable for latency claims.',
        frames=len(frames), overall=summarize(objects),
        groups={name: summarize(rows) for name, rows in groups.items()}, rows=frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(frames=len(frames), overall=result['overall']), indent=2))


if __name__ == '__main__':
    main()
