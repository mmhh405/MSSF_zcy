"""V2 identity, gradients, query entry, clean-anchor routing, and config checks."""
import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from mmdet3d.utils import register_all_modules
from mmdet3d.models.layers.fusion_layers.reliability_gate import ReliabilityGate
from mmdet3d.models.layers.fusion_layers.reliability_gate_v2 import ReliabilityGateV2
from mmdet3d.models.layers.fusion_layers.radar_img_cross_att import MSDeformableAttention
from mmdet3d.models.layers.fusion_layers.query_reliability import gate_query_image
from mmdet3d.datasets.transforms.image_degradation import (
    RandomImageDegradation, PackReliabilityInputs)
from mmdet3d.engine.hooks.reliability_experiment_hook import update_result_files


ROOT = Path(__file__).resolve().parents[1]


class V2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()

    def test_identity_with_different_image_channels(self):
        gate = ReliabilityGateV2(8, image_channels=12)
        radar, image = torch.randn(2, 4, 8), torch.randn(2, 4, 12)
        out, scores = gate(radar, image, torch.ones(2, 4, 1, dtype=torch.bool),
                           torch.zeros(2, 4, 3), return_gate=True)
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))
        self.assertTrue(torch.equal(out, image))

    def test_initial_bias_gradient_exceeds_v1(self):
        gradients = []
        for gate in [ReliabilityGate(8), ReliabilityGateV2(8)]:
            value = torch.ones(1, 2, 8)
            gate(value, value, torch.ones(1, 2, 1, dtype=torch.bool),
                 torch.zeros(1, 2, 3)).sum().backward()
            gradients.append(abs(float(gate.predictor[-1].bias.grad)))
        self.assertGreater(gradients[1] / gradients[0], 50.)

    def test_clean_anchor_selectivity_and_no_prediction_metadata_leak(self):
        gate = ReliabilityGateV2(8)
        gate.predictor[-1].bias.data.fill_(1.)
        radar, image = torch.randn(2, 8), torch.randn(2, 8)
        valid, geometry = torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 3)
        out_clean = gate(radar, image, valid, geometry, clean_mask=valid)
        total, count = gate.consume_clean_anchor()
        self.assertEqual(float(count), 2.)
        self.assertGreater(float(total), 0.)
        total.backward()
        self.assertGreater(float(gate.predictor[-1].bias.grad.abs()), 0.)
        out_corrupt = gate(radar, image, valid, geometry, clean_mask=~valid)
        zero, count = gate.consume_clean_anchor()
        self.assertEqual(float(count), 0.)
        self.assertEqual(float(zero), 0.)
        self.assertTrue(torch.equal(out_clean, out_corrupt))
        self.assertFalse(gate._anchor_sums)
        gate.eval()
        gate(radar, image, valid, geometry, clean_mask=valid)
        self.assertFalse(gate._anchor_sums)

    def test_invalid_projection_and_nonfinite_values(self):
        gate = ReliabilityGateV2(8)
        radar, image = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
        image[0, 1, 0] = float('nan')
        valid = torch.tensor([[[True], [True], [False]]])
        out, scores = gate(radar, image, valid, torch.zeros(1, 3, 3), True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(scores[0, 1:], torch.zeros(2, 1)))
        self.assertTrue(torch.equal(out[0, 2], torch.zeros(8)))

    def test_attention_matches_actual_baseline_at_identity(self):
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            'baseline/mssf-pp-vod-v1:mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py'], text=True)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef): node.decorator_list = []
        namespace = {}
        exec(compile(tree, '<baseline>', 'exec'), namespace)
        cfg = dict(query_embed_dims=8, value_embed_dims=8, output_embed_dims=8,
                   num_heads=2, num_levels=1, num_points=2, dropout=0.)
        original = namespace['MSDeformableAttention'](**cfg).eval()
        revised = MSDeformableAttention(**cfg,
                    reliability_cfg=dict(type='ReliabilityGateV2')).eval()
        missing, unexpected = revised.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith('reliability_gate.') for k in missing))
        query = torch.randn(1, 3, 8)
        args = dict(query=query, value=torch.randn(1, 4, 8),
                    reference_points=torch.full((1, 3, 1, 2), .5),
                    spatial_shapes=torch.tensor([[2, 2]]), level_start_index=torch.tensor([0]))
        actual = revised(**args, radar_features=query,
                         projection_valid=torch.ones(1, 3, 1, dtype=torch.bool),
                         geometry=torch.zeros(1, 3, 3))[0]
        self.assertTrue(torch.equal(actual, original(**args)[0]))

    def test_query_gate_controls_preprojection_image_contribution(self):
        meta = dict(lidar2img=np.eye(4), input_shape=(100, 100),
                    img_shape=(100, 100), transformation_3d_flow=[], v2_is_clean=True)
        gate = ReliabilityGateV2(8, image_channels=12)
        radar, image = torch.randn(2, 8), torch.randn(2, 12)
        centers = torch.tensor([[0., 20., 30., 1.], [0., 40., 50., 1.]])
        coordinates = torch.zeros(2, 4, dtype=torch.long)
        self.assertTrue(torch.equal(
            gate_query_image(gate, radar, image, centers, coordinates, [meta]), image))
        gate.predictor[-1].bias.data.fill_(-10.)
        suppressed = gate_query_image(gate, radar, image, centers, coordinates, [meta])
        self.assertLess(float(suppressed.abs().sum()), float(image.abs().sum()) * .001)

    def test_training_degradation_and_packing_preserve_other_inputs(self):
        np.random.seed(3)
        image = np.full((40, 40, 3), 100, dtype=np.uint8)
        points, labels, projection = object(), object(), np.eye(4)
        data = dict(img=image, points=points, labels=labels, lidar2img=projection)
        clean = RandomImageDegradation(prob=0)(dict(data))
        bad = RandomImageDegradation(prob=1)(dict(data))
        self.assertTrue(clean['v2_is_clean'])
        self.assertFalse(bad['v2_is_clean'])
        self.assertEqual(bad['img'].shape, image.shape)
        self.assertEqual(bad['img'].dtype, image.dtype)
        self.assertTrue(np.array_equal(image, np.full_like(image, 100)))
        self.assertIs(bad['points'], points)
        self.assertIs(bad['labels'], labels)
        self.assertIs(bad['lidar2img'], projection)
        packed = PackReliabilityInputs(keys=['img'])(bad)
        self.assertFalse(packed['data_samples'].metainfo['v2_is_clean'])

    def test_config_ablation_and_v1_pipeline_isolation(self):
        folder = ROOT / 'configs/experiments/reliability_fusion_v2'
        for path in folder.glob('*.py'):
            cfg = Config.fromfile(str(path))
            pipeline = cfg.train_dataloader.dataset.dataset.pipeline
            self.assertEqual(sum(s['type'] == 'RandomImageDegradation' for s in pipeline), 1)
            self.assertEqual(pipeline[-1]['type'], 'PackReliabilityInputs')
            self.assertEqual(cfg.custom_hooks[0].readme, 'README_reliability_fusion_v2.md')
        no_aug = Config.fromfile(str(folder / 'control_clean.py'))
        aug = Config.fromfile(str(folder / 'control_aug.py'))
        self.assertEqual(no_aug.train_dataloader.dataset.dataset.pipeline[2]['prob'], 0.)
        self.assertEqual(aug.train_dataloader.dataset.dataset.pipeline[2]['prob'], .5)
        v1 = Config.fromfile(str(ROOT / 'configs/experiments/reliability_fusion/mssf_pp_vod_control.py'))
        self.assertFalse(any(s['type'] == 'RandomImageDegradation'
                             for s in v1.train_dataloader.dataset.dataset.pipeline))

    def test_v2_result_files_do_not_overwrite_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'README_reliability_fusion.md'
            original.write_text('V1 remains unchanged')
            (root / 'README_reliability_fusion_v2.md').write_text(
                '<!-- RELIABILITY_RESULTS_START --><!-- RELIABILITY_RESULTS_END -->')
            update_result_files(root, dict(run_id='check', variant='v2', status='smoke_only'),
                                 'README_reliability_fusion_v2.md',
                                 'repro/reliability_fusion_v2/results.json')
            self.assertEqual(original.read_text(), 'V1 remains unchanged')
            self.assertEqual(len(json.loads((root / 'repro/reliability_fusion_v2/results.json').read_text())), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
