"""CPU recovery checks; no training or data mutation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location('queue', Path(__file__).resolve().parents[1] /
                                            'tools/analysis_tools/run_v3_prior_repeats.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def checkpoint(path, epoch=5, value=1.):
    torch.save(dict(meta=dict(epoch=epoch, iter=5139 * epoch),
                    state_dict={'weight': torch.tensor([value])},
                    optimizer={}, param_schedulers=[]), path)


class RecoveryChecks(unittest.TestCase):
    def test_reject_truncated_and_nonfinite_keep_latest_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint(run / 'epoch_4.pth', 4)
            checkpoint(run / 'epoch_5.pth', 5)
            (run / 'epoch_6.pth').touch()
            checkpoint(run / 'epoch_3.pth', 3, float('nan'))
            epoch, path = queue.select_resume(run)
            self.assertEqual(epoch, 5)
            self.assertEqual(path.name, 'epoch_5.pth')
            self.assertEqual(len(list((run / 'rejected_checkpoints').glob('*'))), 2)

    def test_zero_byte_checkpoint_is_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'epoch_6.pth').touch()
            self.assertIsNone(queue.completed_result(run, 1))

    def test_final_metrics_require_matching_seed_and_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint(run / 'epoch_6.pth', 6)
            result = dict(summary=dict(training_completed=True, random_seed=1,
                                       validation_epoch=6, iterations=30834,
                                       train_samples=5139, samples=1296,
                                       eaa_map=63., dc_map=82.))
            (run / 'reliability_result.json').write_text(json.dumps(result))
            self.assertEqual(queue.completed_result(run, 1)['eaa_map'], 63.)
            with self.assertRaisesRegex(RuntimeError, 'protocol mismatch'):
                queue.completed_result(run, 2)


if __name__ == '__main__':
    unittest.main()
