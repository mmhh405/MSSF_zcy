"""Measure paired clean/blur feature drift for the three seed-0 VoD models."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import load
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet3d.evaluation.metrics.vod_metric import VODMetric
from mmdet3d.models.layers.fusion_layers.coord_transform import apply_3d_transformation
from mmdet3d.registry import MODELS, METRICS
from mmdet3d.utils import register_all_modules
from reliability_robustness import ROOT, SETTINGS, align_sample_indices, corrupt_batch, file_hash, uniform_indices

WORK = Path('work_dirs/reliability_fusion_v2')
ANN = Path('data/vod/kitti_infos_val.pkl')
MODELS_TO_RUN = {
    'control_aug': WORK / 'control_aug/20261030_214524/epoch_6.pth',
    'gate_main': WORK / 'gate_main/20261030_214524/epoch_6.pth',
    'gate_aug_query_only': WORK / 'ablation/query_only_20261031_201602/gate_aug_query_only/epoch_6.pth',
}


def regions(anno, scale, bounds, meta):
    """Sample the interior of annotated image boxes and oriented BEV footprints."""
    t = np.array([1/6, .5, 5/6])
    grid = np.stack(np.meshgrid(t, t), -1).reshape(9, 2)
    cam = anno['images']['CAM2']
    inverse = np.linalg.inv(cam['lidar2cam'])
    image, bev, targets = [], [], []
    for j, obj in enumerate(anno['instances']):
        if obj['bbox_label'] not in (0, 1, 2):
            continue
        x, y, z, length, height, width, yaw = obj['bbox_3d']
        if not 15 <= z < 30 or obj['bbox'][3] - obj['bbox'][1] <= 40 or obj['occluded'] > 4:
            continue
        lo, hi = np.array(obj['bbox']).reshape(2, 2)
        pixels = (lo + grid * (hi - lo)) * scale
        local = (grid - .5) * [length, width]
        c, s = np.cos(yaw), np.sin(yaw)
        xyz = np.column_stack((x + c*local[:, 0] + s*local[:, 1],
                               np.full(9, y-height/2),
                               z - s*local[:, 0] + c*local[:, 1],
                               np.ones(9)))
        lidar = torch.as_tensor((xyz @ inverse.T)[:, :3].copy(), dtype=torch.float32)
        xy = apply_3d_transformation(lidar, 'LIDAR', meta).numpy()[:, :2]
        unit = (xy - bounds[:2]) / (bounds[3:5] - bounds[:2])
        if not np.all((unit >= 0) & (unit < 1)):
            continue
        image.append(pixels)
        bev.append(unit)
        targets.append(dict(gt_index=j, category=('Pedestrian','Cyclist','Car')[obj['bbox_label']], depth_m=z))
    return targets, np.asarray(image), np.asarray(bev)


def pool(feature, unit):
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f'Expected one NCHW map, got {feature.shape}')
    grid = feature.new_tensor(2 * unit - 1)[None]
    sampled = F.grid_sample(feature.float(), grid.float(), align_corners=False, padding_mode='border')
    return sampled[0].mean(-1).T.cpu()


def drift(clean, blur):
    if clean.shape != blur.shape or not (torch.isfinite(clean).all() and torch.isfinite(blur).all()):
        raise ValueError('Invalid paired features')
    a, b = clean.norm(dim=1), blur.norm(dim=1)
    cosine = 1 - F.cosine_similarity(clean, blur).clamp(-1, 1)
    return [dict(cosine_distance=float(c) if min(x,y)>1e-8 else None,
                 relative_norm_change=float(y/x-1) if x>1e-8 else None)
            for c,x,y in zip(cosine,a,b)]


def save_alignment(path, anno, feature, pixels, bev, scale, padded, bounds):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw = cv2.imread(str(Path('data/vod/training/image_2') / anno['images']['CAM2']['img_path']))
    if raw is None:
        raise ValueError('Missing alignment image')
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    points = (pixels / scale).reshape(-1,2)
    axes[0].scatter(*points.T, s=4, c='lime')
    axes[0].set_title(f'Image ROI samples, padded input {padded}')
    norm = feature[0].float().norm(dim=0).cpu().numpy()
    axes[1].imshow(np.log1p(norm), origin='lower', extent=(bounds[0],bounds[3],bounds[1],bounds[4]))
    points = bev.reshape(-1,2) * (bounds[3:5]-bounds[:2])+bounds[:2]
    axes[1].scatter(*points.T,s=5,c='lime')
    axes[1].set(xlabel='Radar x (m)',ylabel='Radar y (m)',title='Detection neck BEV and ROI samples')
    fig.tight_layout()
    fig.savefig(path,dpi=130)
    plt.close(fig)


def evaluate(name, checkpoint, annotations, indices, out):
    config = Path(f'configs/experiments/reliability_fusion_v2/{name}.py')
    cfg = Config.fromfile(config)
    cfg.test_dataloader.dataset.indices = indices
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    loader = Runner.build_dataloader(cfg.test_dataloader, seed=0)
    model = MODELS.build(cfg.model).cuda().eval()
    digest = file_hash(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    bounds = np.asarray(cfg.model.voxel_encoder.point_cloud_range)
    voxel = np.asarray(cfg.model.voxel_encoder.voxel_size)
    captured = {}
    def hook(key):
        def capture(module, inputs, output):
            if key in captured:
                raise ValueError(f'Duplicate hook {key}')
            captured[key] = output
        return capture
    def input_hook(module, inputs):
        captured['padded'] = tuple(inputs[0].shape[-2:])
    handles = [model.img_backbone.register_forward_pre_hook(input_hook),
               model.img_neck.register_forward_hook(hook('fpn')),
               model.neck.register_forward_hook(hook('bev'))]
    metric_cfg, = cfg.test_evaluator
    metrics = {c: METRICS.build(metric_cfg) for c in ('clean','blur')}
    assert all(isinstance(m, VODMetric) for m in metrics.values())
    rows, frames, diagrams = [], [], 0
    try:
        with torch.inference_mode():
            for offset, batch in enumerate(loader):
                index = indices[offset]
                anno = annotations[index]
                vectors = {}
                layouts = {}
                for condition, metric in metrics.items():
                    captured.clear()
                    view = corrupt_batch(copy.deepcopy(batch), condition)
                    predictions = model.test_step(view)
                    assert len(predictions) == 1
                    align_sample_indices(predictions, indices, offset, annotations)
                    p = predictions[0]
                    assert Path(p.img_path).name == anno['images']['CAM2']['img_path']
                    assert not p.metainfo.get('flip', False)
                    scale = np.asarray(p.metainfo.get('scale_factor', (1., 1.))[:2])
                    assert np.allclose(p.metainfo.get('img_crop_offset', (0., 0.)), 0)
                    cam = anno['images']['CAM2']
                    assert np.allclose(p.img_shape[:2], np.array([cam['height'], cam['width']])*scale[::-1], atol=1)
                    h, w = captured['padded']
                    targets, pixels, bev = regions(anno, scale, bounds, p.metainfo)
                    fpn = captured['fpn']
                    bev_map, = captured['bev']
                    hw = np.array(bev_map.shape[-2:][::-1])
                    stride = (bounds[3:5]-bounds[:2])/voxel[:2]/hw
                    if not np.allclose(stride, 2):
                        raise ValueError(f'Unexpected detection BEV stride {stride}')
                    layouts[condition] = dict(fpn=[list(x.shape) for x in fpn], bev=list(bev_map.shape),
                                              padded=[h,w], scale=scale.tolist(), targets=targets)
                    if targets:
                        assert np.all((pixels >= 0) & (pixels < [w,h]))
                        vectors[condition] = [pool(level,pixels/[w,h]) for level in fpn]+[pool(bev_map,bev)]
                        if name == 'gate_main' and condition == 'clean' and diagrams < 3 and any(t['category']=='Pedestrian' for t in targets):
                            save_alignment(out/f'alignment-{index}.png',anno,bev_map,pixels,bev,scale,(h,w),bounds)
                            diagrams += 1
                    metric.process(view,[p.to_dict()])
                assert layouts['clean'] == layouts['blur']
                if targets:
                    values = [drift(a,b) for a,b in zip(vectors['clean'],vectors['blur'])]
                    for j, target in enumerate(targets):
                        rows.append(dict(frame=index,**target,fpn=[v[j] for v in values[:-1]],bev=values[-1][j]))
                frames.append(index)
                if (offset+1)%32 == 0:
                    print(name,offset+1,len(indices),flush=True)
    finally:
        for handle in handles:
            handle.remove()
    assert frames == indices and file_hash(checkpoint) == digest
    dest = out/name
    dest.mkdir()
    for condition, metric in metrics.items():
        metric.data_infos = metric.convert_annos_to_kitti_annos(load(ANN))
        with torch.inference_mode():
            result, _ = metric.format_results(metric.results,pklfile_prefix=str(dest/condition),
                                             classes=loader.dataset.metainfo['classes'])
        assert len(result['pred_instances_3d']) == len(indices)
    (dest/'objects.json').write_text(json.dumps(rows,allow_nan=False),encoding='utf-8')
    return dict(objects=len(rows), pedestrian=sum(r['category']=='Pedestrian' for r in rows),
                checkpoint=str(checkpoint),sha256=digest, config_sha256=file_hash(config),
                alignment_diagrams=diagrams, fpn_shapes=layouts['clean']['fpn'],
                bev_shape=layouts['clean']['bev'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--work-dir',type=Path,required=True)
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    with (WORK/'.gpu0-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        annotations = load(ANN)['data_list']
        indices = uniform_indices(len(annotations),args.samples)
        args.work_dir.mkdir(parents=True,exist_ok=False)
        register_all_modules()
        result = {}
        for name, checkpoint in MODELS_TO_RUN.items():
            result[name] = evaluate(name,checkpoint,annotations,indices,args.work_dir)
            torch.cuda.empty_cache()
        keys = []
        for name in result:
            rows = json.loads((args.work_dir/name/'objects.json').read_text())
            keys.append([(row['frame'], row['gt_index'], row['category']) for row in rows])
        assert all(k == keys[0] for k in keys[1:])
        assert all(x['pedestrian'] for x in result.values())
        (args.work_dir/'summary.json').write_text(json.dumps(dict(indices=indices,models=result,settings=SETTINGS, annotation_sha256=file_hash(ANN),script_sha256=file_hash(__file__)),indent=2),encoding='utf-8')
        print('COMPLETED',args.work_dir/'summary.json')


if __name__ == '__main__':
    main()
