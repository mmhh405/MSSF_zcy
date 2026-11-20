"""Regression checks for opt-in spatial coverage with unchanged legacy routing."""
import copy
import subprocess
import types
import math
import unittest
from pathlib import Path

import torch
from mmengine.config import Config

from mmdet3d.models.layers.fusion_layers.early_budget_gate_v3 import (
    BidirectionalBudgetRouter, select_image_with_coverage)
from test_reliability_fusion_v3 import inputs


class SpatialCoverageTest(unittest.TestCase):
    def test_exact_budget_for_rectangles_and_small_budgets(self):
        for rows, cols in [(1, 1), (1, 9), (9, 1), (3, 5), (5, 8)]:
            scores = torch.arange(rows * cols).float()
            for ratio in [.01, .2, .5, 1.]:
                for policy in ['index', 'spatial']:
                    keep = select_image_with_coverage(
                        scores, scores.flip(0), scores, (rows, cols), ratio,
                        coverage=policy)
                    self.assertEqual(int(keep.sum()), math.ceil(len(scores) * ratio))
        empty = torch.empty(0)
        self.assertEqual(select_image_with_coverage(
            empty, empty, empty, (0, 0), .2).numel(), 0)

    def test_spatial_reserve_reaches_far_tile(self):
        joint = torch.arange(15).float()
        own, cross = torch.zeros(15), torch.zeros(15)
        own[0], cross[1] = 100., 100.
        spatial = select_image_with_coverage(joint, own, cross, (3, 5), .2)
        index = select_image_with_coverage(
            joint, own, cross, (3, 5), .2, coverage='index')
        self.assertEqual(spatial.nonzero().flatten().tolist(), [0, 1, 14])
        self.assertEqual(index.nonzero().flatten().tolist(), [0, 1, 2])

    def test_legacy_forward_matches_archived_source(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            ['git', 'show', 'f5c9726:mmdet3d/models/layers/fusion_layers/early_budget_gate_v3.py'],
            cwd=root, text=True)
        archived = types.ModuleType('mmdet3d.models.layers.fusion_layers._archived_v3')
        exec(compile(source, '<archived_v3>', 'exec'), archived.__dict__)
        data, metas = inputs(1)
        torch.manual_seed(7)
        old = archived.BidirectionalBudgetRouter(tile_size=64).eval()
        new = BidirectionalBudgetRouter(tile_size=64).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            a = old(data['imgs'], data['voxels'], copy.deepcopy(metas))
            b = new(data['imgs'], data['voxels'], copy.deepcopy(metas))
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in ['image_mask', 'radar_mask', 'visual_logits',
                    'image_logits', 'radar_logits']:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_spatial_router_keeps_radar_and_ignores_label_metadata(self):
        data, metas = inputs(2)
        router = BidirectionalBudgetRouter(
            tile_size=64, radar_keep_ratio=1., image_selection='spatial').eval()
        altered = copy.deepcopy(metas)
        for meta in altered:
            meta.update(fake_gt='unused', v2_is_clean=False)
        with torch.no_grad():
            a = router(data['imgs'], data['voxels'], copy.deepcopy(metas))
            b = router(data['imgs'], data['voxels'], altered)
        self.assertTrue(a['radar_mask'].all())
        self.assertTrue(torch.equal(a['image_mask'], b['image_mask']))
        expected = math.ceil(a['image_mask'][0].numel() * .2)
        self.assertEqual(a['image_mask'].sum((1, 2)).tolist(), [expected, expected])

    def test_projected_coverage_for_full_and_empty_selection(self):
        from types import SimpleNamespace
        from mmdet3d.structures import LiDARInstance3DBoxes
        from tools.analysis_tools.diagnose_v31_tiles import projected_box_coverage
        _, metas = inputs(1)
        sample = SimpleNamespace(gt_instances_3d=SimpleNamespace(
            labels_3d=torch.tensor([0]),
            bboxes_3d=LiDARInstance3DBoxes(
                torch.tensor([[1., 1., 1., .2, .2, .2, 0.]]))))
        for selected, expected in [(True, 1.), (False, 0.)]:
            records, excluded = projected_box_coverage(
                torch.full((2, 3), selected), (256, 384), sample, metas[0], ['Car'])
            self.assertEqual(excluded, 0)
            self.assertEqual(len(records), 1)
            self.assertAlmostEqual(records[0]['selected_area_fraction'], expected, places=6)
            self.assertEqual(records[0]['centre_selected'], selected)

    def test_configs_share_training_and_isolate_outputs(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'configs/experiments/reliability_fusion_v31'
        configs = {p.stem: Config.fromfile(str(p)) for p in folder.glob('*.py')}
        self.assertEqual(len(configs), 5)
        self.assertEqual(len({c.work_dir for c in configs.values()}), 5)
        for cfg in configs.values():
            self.assertEqual(cfg.load_from, configs['legacy'].load_from)
            self.assertTrue(Path(cfg.load_from).is_file())
            self.assertEqual(cfg.train_cfg.max_epochs, 6)
            self.assertEqual(cfg.model.early_budget.router.radar_keep_ratio, 1.)
            self.assertEqual(cfg.model.early_budget.router.image_keep_ratio, .2)
            self.assertEqual(cfg.model.early_budget.image_executor.global_scale, .625)
            self.assertEqual(cfg.model.test_cfg.nms_pre, 300)
            self.assertEqual(cfg.model.test_cfg.score_thr, .1)
            self.assertEqual(cfg.custom_hooks, [])
        self.assertEqual(configs['spatial'].model.early_budget.router.image_selection,
                         'spatial')
        independent = configs['spatial_independent'].model.early_budget.router
        self.assertFalse(independent.use_radar_to_image)
        self.assertFalse(independent.use_image_to_radar)


if __name__ == '__main__':
    unittest.main(verbosity=2)
