"""Targeted checks for gating, projection masks, residuals, and README recording."""
import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mmdet3d.utils import register_all_modules
from mmdet3d.models.layers.fusion_layers.reliability_gate import ReliabilityGate
from mmdet3d.models.layers.fusion_layers.radar_img_cross_att import (
    MSDeformableAttention, RadarImageCrossAttention)
from mmdet3d.models.layers.fusion_layers.voxel_fusion_block import VoxelFusionBlock
from mmdet3d.engine.hooks.reliability_experiment_hook import update_result_files
import mmdet3d.engine.hooks.reliability_experiment_hook as result_module
from mmengine.config import Config


class ReliabilityFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()

    def test_invalid_projection_and_nonfinite_update_are_suppressed(self):
        gate = ReliabilityGate(8, hidden_channels=4)
        radar, image = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        image[0, 1, 0] = float('nan')
        valid = torch.ones(2, 3, 1, dtype=torch.bool)
        valid[1, 2] = False
        output, scores = gate(radar, image, valid, torch.zeros(2, 3, 3), True)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(8)))
        self.assertTrue(torch.equal(output[1, 2], torch.zeros(8)))
        self.assertAlmostEqual(scores[0, 0].item(), .99, places=5)

    def test_gate_has_trainable_gradients(self):
        torch.manual_seed(17)
        gate = ReliabilityGate(8, hidden_channels=4)
        optimizer = torch.optim.SGD(gate.parameters(), lr=.1)
        radar, image = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        valid = torch.ones(2, 4, 1, dtype=torch.bool)
        for _ in range(3):
            optimizer.zero_grad()
            output = gate(radar, image, valid, torch.zeros(2, 4, 3))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(gate.predictor[-1].weight.grad).all())
            self.assertGreater(gate.predictor[-1].weight.grad.abs().sum().item(), 0.)
            optimizer.step()
        self.assertGreater(gate.radar_proj.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(gate.image_proj.weight.grad.abs().sum().item(), 0.)

    def test_gate_config_does_not_leak_into_other_instances(self):
        cfg = dict(query_embed_dims=8, value_embed_dims=8, output_embed_dims=8)
        enabled = RadarImageCrossAttention(**cfg, reliability_cfg=dict(hidden_channels=4))
        baseline = RadarImageCrossAttention(**cfg)
        self.assertIsNotNone(enabled.deformable_attention.reliability_gate)
        self.assertIsNone(baseline.deformable_attention.reliability_gate)

    def test_projection_depth_bounds_and_padding(self):
        meta = dict(lidar2img=np.eye(4), input_shape=(100, 100),
                    img_shape=(80, 90), transformation_3d_flow=[])
        points = torch.tensor([[50., 50., 1.], [95., 50., 1.],
                               [50., 85., 1.], [0., 0., -1.], [-1., 50., 1.]])
        valid, geometry = VoxelFusionBlock.projection_context(points, meta)
        self.assertEqual(valid.flatten().tolist(), [True, False, False, False, False])
        self.assertEqual(tuple(geometry.shape), (1, 5, 3))

    def test_disabled_gate_keeps_state_and_forward_identical(self):
        # Compare against the actual frozen baseline source, not another copy
        # of the modified class. Remove registration decorators in this namespace.
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output([
            'git', '-C', str(root), 'show',
            'baseline/mssf-pp-vod-v1:mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py'],
            text=True)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
        namespace = {}
        exec(compile(tree, '<frozen-baseline-attention>', 'exec'), namespace)
        cfg = dict(query_embed_dims=8, value_embed_dims=8, output_embed_dims=8,
                   num_heads=2, num_levels=1, num_points=2, dropout=0.)
        original = namespace['MSDeformableAttention'](**cfg).eval()
        disabled = MSDeformableAttention(**cfg, reliability_cfg=dict(enabled=False)).eval()
        disabled.load_state_dict(original.state_dict(), strict=True)
        args = dict(query=torch.randn(1, 3, 8), value=torch.randn(1, 4, 8),
                    reference_points=torch.full((1, 3, 1, 2), .5),
                    spatial_shapes=torch.tensor([[2, 2]]), level_start_index=torch.tensor([0]))
        self.assertEqual(set(original.state_dict()), set(disabled.state_dict()))
        self.assertTrue(torch.equal(original(**args)[0], disabled(**args)[0]))
        enabled = MSDeformableAttention(**cfg, reliability_cfg=dict(hidden_channels=4)).eval()
        missing, unexpected = enabled.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith('reliability_gate.') for k in missing))
        result = enabled(**args, radar_features=args['query'],
                         projection_valid=torch.zeros(1, 3, 1, dtype=torch.bool),
                         geometry=torch.zeros(1, 3, 3))[0]
        self.assertTrue(torch.equal(result, args['query']))

    def test_result_upsert_preserves_readme_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readme = root / 'README_reliability_fusion.md'
            readme.write_text('前言\n<!-- RELIABILITY_RESULTS_START -->\n'
                              '<!-- RELIABILITY_RESULTS_END -->\n后记', encoding='utf-8')
            entry = dict(run_id='one', variant='test', status='smoke_only',
                         eaa_map=0., dc_map=None, git_commit='abcdef0123',
                         git_dirty_at_start=True)
            update_result_files(root, entry)
            entry['eaa_map'] = 12.5
            update_result_files(root, entry)
            rows = json.loads((root / 'repro/reliability_fusion/results.json').read_text())
            self.assertEqual(len(rows), 1)
            text = readme.read_text(encoding='utf-8')
            self.assertTrue(text.startswith('前言') and text.endswith('后记'))
            self.assertIn('12.5000', text)
            self.assertIn('smoke_only', text)
            self.assertIn('abcdef01+dirty', text)
            self.assertIn('—', text)

    def test_validation_then_training_completion_keeps_actual_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README_reliability_fusion.md').write_text(
                '<!-- RELIABILITY_RESULTS_START --><!-- RELIABILITY_RESULTS_END -->')
            train_loop = SimpleNamespace(dataloader=SimpleNamespace(dataset=[1, 2]))
            runner = SimpleNamespace(
                cfg=Config(dict(reliability_experiment=dict(variant='integration', purpose='smoke'))),
                work_dir=str(root / 'run'), timestamp='test', seed=0, epoch=1, iter=2,
                _train_loop=train_loop, train_loop=train_loop,
                val_loop=SimpleNamespace(dataloader=SimpleNamespace(dataset=list(range(16)))),
                message_hub=SimpleNamespace(get_info=lambda key: None,
                    get_scalar=lambda key: SimpleNamespace(current=lambda: 1.25)))
            with patch.object(result_module, 'ROOT', root), patch.object(
                    result_module.subprocess, 'check_output', side_effect=['a' * 40, '']):
                hook = result_module.ReliabilityExperimentHook()
                hook.before_run(runner)
                hook.after_val_epoch(runner, {'x/entire_area/map': 12.5, 'x/roi/map': 20.})
                hook.after_train(runner)
            entry = json.loads((root / 'repro/reliability_fusion/results.json').read_text())[0]
            self.assertEqual(entry['eaa_map'], 12.5)
            self.assertEqual(entry['dc_map'], 20.)
            self.assertEqual(entry['validation_epoch'], 1)
            self.assertEqual(entry['samples'], 16)
            self.assertEqual(entry['train_samples'], 2)
            self.assertEqual(entry['last_logged_train_loss'], 1.25)
            self.assertEqual(entry['status'], 'smoke_only')
            self.assertTrue(entry['training_completed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
