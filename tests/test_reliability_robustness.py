"""Check deterministic image-only changes used by the diagnostic experiment."""
import importlib.util
import unittest
from pathlib import Path

import torch
from mmdet3d.structures import Det3DDataSample

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'robustness', ROOT / 'tools/analysis_tools/reliability_robustness.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CorruptionTest(unittest.TestCase):
    def test_clean_and_dark(self):
        image = torch.full((3, 40, 40), 100, dtype=torch.uint8)
        self.assertIs(module.corrupt_image(image, 'clean'), image)
        dark = module.corrupt_image(image, 'dark')
        self.assertTrue(torch.equal(dark, torch.full_like(image, 20)))
        self.assertTrue(torch.equal(image, torch.full_like(image, 100)))

    def test_occlusion_and_blackout(self):
        image = torch.ones(3, 40, 40, dtype=torch.uint8)
        occluded = module.corrupt_image(image, 'occlusion')
        self.assertEqual(int((occluded == 0).sum()), 3 * 20 * 20)
        self.assertEqual(int(module.corrupt_image(image, 'blackout').sum()), 0)
        self.assertEqual(int(image.sum()), 3 * 40 * 40)

    def test_blur_preserves_shape_dtype_and_reduces_variation(self):
        torch.manual_seed(0)
        image = torch.randint(0, 256, (3, 40, 40), dtype=torch.uint8)
        blurred = module.corrupt_image(image, 'blur')
        self.assertEqual(blurred.shape, image.shape)
        self.assertEqual(blurred.dtype, image.dtype)
        self.assertLess(float(blurred.float().std()), float(image.float().std()))
        self.assertTrue(torch.equal(blurred, module.corrupt_image(image, 'blur')))

    def test_points_labels_and_geometry_untouched(self):
        image = torch.ones(3, 40, 40, dtype=torch.uint8)
        points, labels, geometry = torch.randn(8, 7), object(), object()
        data = dict(inputs=dict(img=[image], points=[points]),
                    data_samples=[labels], geometry=geometry)
        result = module.corrupt_batch(data, 'blackout')
        self.assertIs(result['inputs']['points'][0], points)
        self.assertIs(result['data_samples'][0], labels)
        self.assertIs(result['geometry'], geometry)
        self.assertEqual(int(result['inputs']['img'][0].sum()), 0)
        self.assertEqual(int(image.sum()), 3 * 40 * 40)

    def test_uniform_selection_is_identical_for_both_models(self):
        indices = module.uniform_indices(1296, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual((indices[0], indices[-1]), (0, 1295))
        self.assertEqual(module.uniform_indices(1296, 1296), list(range(1296)))
        with self.assertRaises(ValueError):
            module.uniform_indices(1296, 1297)

    def test_subset_indices_restore_original_annotation_mapping(self):
        annotations = [dict(lidar_points=dict(lidar_path=f'{i:05d}.bin')) for i in range(3)]
        sample = Det3DDataSample(metainfo=dict(sample_idx=0, lidar_path='/data/00002.bin'))
        offset = module.align_sample_indices([sample], [2], 0, annotations)
        self.assertEqual(offset, 1)
        self.assertEqual(sample.sample_idx, 2)
        wrong = Det3DDataSample(metainfo=dict(sample_idx=0, lidar_path='/data/00001.bin'))
        with self.assertRaisesRegex(ValueError, 'Frame/annotation mismatch'):
            module.align_sample_indices([wrong], [2], 0, annotations)


if __name__ == '__main__':
    unittest.main(verbosity=2)
