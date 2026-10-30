"""Validate MSSF inference dependencies without training or a dataset."""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ['torch','torchvision','mmcv','mmengine','mmdet','mmdet3d','spconv-cu118','cumm-cu118','torch-scatter','numpy','numba','llvmlite','scipy']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'repro/mssf-bench/verification.json'))
    args = parser.parse_args()
    report = {'python': platform.python_version(), 'packages': {}, 'checks': {}}
    for package in PACKAGES:
        report['packages'][package] = importlib.metadata.version(package)
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    report['gpu'] = torch.cuda.get_device_name(0)
    report['cuda_runtime'] = torch.version.cuda
    report['gpu_capability'] = list(torch.cuda.get_device_capability(0))

    def check(name, function):
        started = time.monotonic()
        result = function()
        torch.cuda.synchronize()
        report['checks'][name] = {'passed': True, 'seconds': round(time.monotonic() - started, 3), 'detail': result}
        print('PASS', name, report['checks'][name], flush=True)

    def matmul():
        x = torch.randn(128, 128, device='cuda')
        assert torch.isfinite(x @ x).all()
        return 'CUDA tensor multiplication'

    def voxelization():
        from mmcv.ops import Voxelization
        points = torch.tensor([[1., 0., 0., 1., 0., 0., 0.], [1.01, 0., 0., 2., 0., 0., 0.]], device='cuda')
        layer = Voxelization(voxel_size=[0.16, 0.16, 5.], point_cloud_range=[0., -25.6, -3., 51.2, 25.6, 2.], max_num_points=32, max_voxels=(16000, 40000))
        voxels, coordinates, counts = layer(points)
        assert voxels.shape[0] == 1 and int(counts[0]) == 2
        return {'voxels': list(voxels.shape)}

    def deformable_attention():
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
        value = torch.randn(1, 4, 1, 8, device='cuda')
        shapes = torch.tensor([[2, 2]], dtype=torch.long, device='cuda')
        starts = torch.tensor([0], dtype=torch.long, device='cuda')
        locations = torch.full((1, 2, 1, 1, 2, 2), 0.5, device='cuda')
        weights = torch.full((1, 2, 1, 1, 2), 0.5, device='cuda')
        out = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 64)
        assert out.shape == (1, 2, 8) and torch.isfinite(out).all()
        return {'output': list(out.shape)}

    def sparse_convolution():
        import spconv.pytorch as spconv
        indices = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2]], dtype=torch.int32, device='cuda')
        tensor = spconv.SparseConvTensor(torch.randn(2, 4, device='cuda'), indices, [4, 4, 4], 1)
        layer = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).cuda().eval()
        out = layer(tensor).features
        assert out.shape == (2, 8) and torch.isfinite(out).all()
        return {'output': list(out.shape)}

    def scatter():
        from torch_scatter import scatter_max
        values = torch.tensor([1., 3., 2.], device='cuda')
        index = torch.tensor([0, 0, 1], device='cuda')
        out, _ = scatter_max(values, index)
        assert torch.equal(out, torch.tensor([3., 2.], device='cuda'))
        return 'torch-scatter CUDA reduction'

    def registration():
        from mmdet3d.utils import register_all_modules
        register_all_modules()
        return 'All MSSF modules registered'

    def rotated_iou():
        from mmdet3d.evaluation.functional.kitti_utils.rotate_iou import rotate_iou_gpu_eval
        boxes = np.array([[0., 0., 2., 2., 0.]], dtype=np.float32)
        overlap = rotate_iou_gpu_eval(boxes, boxes)
        assert np.allclose(overlap, [[1.]], atol=1e-4), overlap
        return {'self_iou': float(overlap[0, 0])}

    def model_checkpoint():
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        from mmdet3d.registry import MODELS
        config = ROOT / 'exps/mssf_pp_vod/pointpillars_hv_secfpn_8xb6-160e_vod-3d-3class_mm_bb_bevfeat_lr0001.py'
        checkpoint = ROOT / 'exps/mssf_pp_vod/epoch_40.pth'
        cfg = Config.fromfile(str(config))
        model = MODELS.build(cfg.model).cuda().eval()
        load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
        count = sum(p.numel() for p in model.parameters())
        del model
        torch.cuda.empty_cache()
        return {'parameters': count, 'checkpoint': str(checkpoint), 'strict_load': True}

    with torch.no_grad():
        for name, function in [('torch_cuda', matmul), ('mmcv_voxelization', voxelization), ('mmcv_deformable_attention', deformable_attention), ('spconv_cuda', sparse_convolution), ('torch_scatter_cuda', scatter), ('mssf_registration', registration), ('numba_rotated_iou', rotated_iou), ('mssf_pp_checkpoint', model_checkpoint)]:
            check(name, function)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('ALL_ENVIRONMENT_CHECKS_PASSED', output, flush=True)


if __name__ == '__main__':
    main()
