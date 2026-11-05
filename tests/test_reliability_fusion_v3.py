"""Execution budgets, geometry, missing modalities, training and storage checks."""
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config

from mmdet3d.models.layers.fusion_layers.early_budget_gate_v3 import (
    BidirectionalBudgetRouter, CoarseTileImageExecutor, points_in_boxes,
    project, select_with_reserves)
from mmdet3d.structures import LiDARInstance3DBoxes


def inputs(batch=2, count=12):
    voxels, coors, points, metas = [], [], [], []
    for b in range(batch):
        # Projection x/z and y/z is inside a 256x384 image.
        p = torch.zeros(count, 7)
        p[:, 0] = torch.arange(count) * .16 + 1
        p[:, 1], p[:, 2], p[:, 3], p[:, 5] = 1., 1., 10., 2.
        coor = torch.zeros(count, 4, dtype=torch.long)
        coor[:, 0] = b
        coor[:, 2] = ((p[:, 1] + 25.6) / .16).floor().long()
        coor[:, 3] = (p[:, 0] / .16).floor().long()
        voxels.append(p[:, None].repeat(1, 2, 1))
        coors.append(coor)
        points.append(p)
        metas.append(dict(lidar2img=np.array([[40., 0., 0., 0.],
            [0., 40., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            input_shape=(256, 384), img_shape=(256, 384), transformation_3d_flow=[]))
    return dict(imgs=torch.randn(batch, 3, 256, 384), points=points,
        voxels=dict(voxels=torch.cat(voxels), coors=torch.cat(coors),
                    num_points=torch.full((batch*count,), 2, dtype=torch.long))), metas


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image):
        self.calls.append(tuple(image.shape))
        return tuple(F.interpolate(image[:, :1],
            (math.ceil(image.shape[-2] / (4*2**i)), math.ceil(image.shape[-1] / (4*2**i))),
            mode='nearest') for i in range(5))


class V3Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_exact_budget_and_three_independent_reserves(self):
        joint = torch.arange(20).float()
        own, cross = torch.zeros(20), torch.zeros(20)
        own[3], cross[7] = 100., 100.
        keep = select_with_reserves(joint, own, cross, .4)
        self.assertEqual(int(keep.sum()), 8)
        self.assertTrue(keep[3] and keep[7] and keep[0])
        self.assertTrue(select_with_reserves(joint, own, cross, 1).all())
        self.assertEqual(select_with_reserves(joint[:0], own[:0], cross[:0], .2).numel(), 0)

    def test_per_sample_radar_budget_and_raw_point_alignment(self):
        data, metas = inputs()
        original = copy.deepcopy(data)
        router = BidirectionalBudgetRouter(min_pillars=1, radar_keep_ratio=.5,
                                           tile_size=128, image_keep_ratio=.5)
        state = router(data['imgs'], data['voxels'], metas)
        routed = router.filter_inputs(data, state)
        for b in range(2):
            self.assertEqual(int(((state['coors'][:, 0] == b) & state['radar_mask']).sum()), 6)
            self.assertEqual(len(routed['points'][b]), 6)
            self.assertTrue(torch.equal(data['points'][b], original['points'][b]))
        self.assertEqual(len(routed['voxels']['voxels']), 12)
        self.assertTrue(torch.equal(data['voxels']['voxels'], original['voxels']['voxels']))

    def test_augmented_projection_and_behind_camera(self):
        _, metas = inputs(1)
        meta = metas[0]
        meta.update(scale_factor=np.array([2., 2.]), flip=True)
        uvz, valid = project(torch.tensor([[1., 1., 1.], [1., 1., -1.]]), meta)
        self.assertAlmostEqual(float(uvz[0, 0]), (384-80)/384, places=6)
        self.assertAlmostEqual(float(uvz[0, 1]), 80/256, places=6)
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])

    def test_metadata_cannot_change_routing(self):
        data, metas = inputs(1)
        router = BidirectionalBudgetRouter(min_pillars=1).eval()
        changed = copy.deepcopy(metas)
        changed[0].update(v2_is_clean=False, v2_degradation='occlusion', fake_gt='unused')
        a = router(data['imgs'], data['voxels'], metas)
        b = router(data['imgs'], data['voxels'], changed)
        for key in ('image_mask', 'radar_mask', 'image_logits', 'radar_logits'):
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_missing_camera_evidence_keeps_radar_budget(self):
        data, metas = inputs(1)
        data['imgs'].zero_()
        metas[0]['lidar2img'][2, 2] = -1
        router = BidirectionalBudgetRouter(min_pillars=1, radar_keep_ratio=.5)
        state = router(data['imgs'], data['voxels'], metas)
        self.assertFalse(state['projections'][0][2].any())
        self.assertEqual(int(state['radar_mask'].sum()), 6)
        self.assertTrue(torch.isfinite(state['radar_logits']).all())

    def test_disabled_image_to_radar_is_independent_of_image(self):
        data, metas = inputs(1)
        router = BidirectionalBudgetRouter(min_pillars=1, use_image_to_radar=False).eval()
        with torch.no_grad():
            router.radar_head[-1].weight.fill_(.1)
        a = router(data['imgs'], data['voxels'], metas)
        b = router(torch.zeros_like(data['imgs']), data['voxels'], metas)
        self.assertTrue(torch.equal(a['radar_logits'], b['radar_logits']))
        self.assertTrue(torch.equal(a['radar_mask'], b['radar_mask']))

    def test_disabled_radar_to_image_is_independent_of_radar(self):
        data, metas = inputs(1)
        router = BidirectionalBudgetRouter(use_radar_to_image=False).eval()
        with torch.no_grad():
            router.image_head.weight.fill_(.1)
        a = router(data['imgs'], data['voxels'], metas)
        changed = copy.deepcopy(data['voxels'])
        changed['voxels'][:, :, 5] += 100
        b = router(data['imgs'], changed, metas)
        self.assertTrue(torch.equal(a['image_logits'], b['image_logits']))
        self.assertTrue(torch.equal(a['image_mask'], b['image_mask']))

    def test_empty_radar_router_and_nonfinite_padding(self):
        data, metas = inputs(1)
        # Ignore nonfinite data outside each pillar's counted slots.
        data['voxels']['voxels'][:, 1] = float('nan')
        data['voxels']['num_points'].fill_(1)
        router = BidirectionalBudgetRouter()
        state = router(data['imgs'], data['voxels'], metas)
        self.assertTrue(torch.isfinite(state['means']).all())
        for key in data['voxels']:
            data['voxels'][key] = data['voxels'][key][:0]
        state = router(data['imgs'], data['voxels'], metas)
        self.assertEqual(state['radar_mask'].numel(), 0)
        self.assertTrue(torch.isfinite(state['image_logits']).all())

    def test_supervision_gradients_and_rotated_box_targets(self):
        data, metas = inputs(1)
        router = BidirectionalBudgetRouter(min_pillars=1)
        boxes = LiDARInstance3DBoxes(torch.tensor([[1.5, 1., 0., 3., 2., 2., .4]]))
        samples = [SimpleNamespace(gt_instances_3d=SimpleNamespace(
            bboxes_3d=boxes, labels_3d=torch.tensor([0])))]
        self.assertTrue(points_in_boxes(torch.tensor([[1.5, 1., 1.]]), boxes).bool().all())
        self.assertFalse(points_in_boxes(torch.tensor([[10., 10., 1.]]), boxes).bool().any())
        optimizer = torch.optim.Adam(router.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            state = router(data['imgs'], data['voxels'], metas)
            fresh_metas = [{k: v for k, v in meta.items() if k != 'input_shape'} for meta in metas]
            losses = router.auxiliary_losses(state, samples, fresh_metas)
            self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
            sum(losses.values()).backward()
            self.assertGreater(float(router.radar_head[-1].weight.grad.abs().sum()), 0)
            optimizer.step()
        self.assertGreater(float(router.preview[0].weight.grad.abs().sum()), 0)

    def test_executor_only_calls_selected_tiles_and_bounds_storage(self):
        image = torch.ones(1, 3, 1216, 1952)
        mask = torch.zeros(1, 5, 8, dtype=torch.bool)
        mask.view(-1)[[0, 5, 10, 15, 20, 25, 30, 39]] = True
        backbone = MockBackbone()
        output, stats = CoarseTileImageExecutor()(image, mask, backbone, None)
        self.assertEqual(sum(shape[0] for shape in backbone.calls), 9)
        self.assertEqual(stats['image_selected_tiles'], 8)
        self.assertLess(stats['image_pixel_work_ratio'], .65)
        self.assertLess(stats['fpn_storage_ratio'], .27)
        self.assertEqual(stats['fpn_persistent_bytes'], sum(x.numel()*x.element_size() for x in output))
        self.assertTrue(torch.equal(image, torch.ones_like(image)))
        self.assertTrue(all(torch.isfinite(x).all() for x in output))
        self.assertTrue(torch.allclose(output[0][:, :, 10:20, 10:20],
                                       torch.ones_like(output[0][:, :, 10:20, 10:20])))

    def test_executor_avoids_unselected_high_resolution_calls(self):
        backbone = MockBackbone()
        _, stats = CoarseTileImageExecutor()(torch.randn(2, 3, 256, 384),
            torch.zeros(2, 2, 3, dtype=torch.bool), backbone, None)
        self.assertEqual(len(backbone.calls), 1)
        self.assertEqual(backbone.calls[0], (2, 3, 128, 192))
        self.assertEqual(stats['image_selected_tiles'], 0)

    def test_standalone_eval_freezes_image_without_weight_initialization(self):
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        register_all_modules()
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(str(root / 'configs/experiments/reliability_fusion_v3/gate_main.py'))
        model = MODELS.build(cfg.model)
        # Do not call init_weights: this matches Runner.test's setup path.
        model.eval()
        for branch in (model.img_backbone, model.img_neck):
            self.assertTrue(all(not p.requires_grad for p in branch.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.early_router.parameters()))
        model.train()
        self.assertTrue(all(not p.requires_grad for p in model.img_backbone.parameters()))

    def test_configurations_keep_protocol_and_result_paths(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'configs/experiments/reliability_fusion_v3'
        for path in folder.glob('*.py'):
            cfg = Config.fromfile(str(path))
            self.assertEqual(cfg.model.type, 'VoxelNetFuseV3')
            self.assertEqual(cfg.custom_hooks[0].readme, 'README_reliability_fusion_v3.md')
            self.assertEqual(cfg.model.test_cfg.nms_pre, 300)
            self.assertEqual(cfg.model.test_cfg.score_thr, .1)
            self.assertEqual(cfg.train_dataloader.dataset.dataset.pipeline[2]['prob'], .5)
        cfg = Config.fromfile(str(folder / 'dense_control.py'))
        self.assertEqual(cfg.model.early_budget.mode, 'dense')
        cfg = Config.fromfile(str(folder / 'independent.py'))
        self.assertFalse(cfg.model.early_budget.router.use_image_to_radar)
        self.assertFalse(cfg.model.early_budget.router.use_radar_to_image)


if __name__ == '__main__':
    unittest.main(verbosity=2)
