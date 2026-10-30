"""Record this experiment's validation/test results in its dedicated README."""
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from mmengine.dist import is_main_process
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


ROOT = Path(__file__).resolve().parents[3]
START = '<!-- RELIABILITY_RESULTS_START -->'
END = '<!-- RELIABILITY_RESULTS_END -->'


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def update_result_files(root, entry):
    """Upsert a run's latest result; full epoch history remains in the run folder."""
    root = Path(root)
    ledger = root / 'repro/reliability_fusion/results.json'
    readme = root / 'README_reliability_fusion.md'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ledger) + '.lock'):
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        records = [row for row in records if row['run_id'] != entry['run_id']]
        records.append(entry)
        header = ['| 实验 | 类型/状态 | 样本数 | 验证 epoch | EAA mAP (%) | DC mAP (%) | 代码 | 运行目录 |',
                  '|---|---|---:|---:|---:|---:|---|---|']
        def cell(value):
            return str(value).replace('|', '\\|').replace('\n', ' ')
        for row in records:
            def metric(name):
                value = row.get(name)
                return '—' if value is None else f'{value:.4f}'
            values = [row.get('variant', row['run_id']), row.get('status', 'unknown'),
                      row.get('samples', '—'), row.get('validation_epoch', '—'),
                      metric('eaa_map'), metric('dc_map'),
                      row.get('git_commit', 'unknown')[:8]
                      + ('+dirty' if row.get('git_dirty_at_start') else ''),
                      row.get('work_dir', '—')]
            header.append('| ' + ' | '.join(map(cell, values)) + ' |')
        original = readme.read_text(encoding='utf-8')
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError('Experiment README must contain one results marker pair')
        prefix, remainder = original.split(START)
        _, suffix = remainder.split(END)
        _atomic_write(ledger, json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        _atomic_write(readme, prefix + START + '\n\n' + '\n'.join(header)
                      + '\n\n' + END + suffix)


@HOOKS.register_module()
class ReliabilityExperimentHook(Hook):
    """Persist actual metrics after validation/test, and mark training completion.

    A table row records the latest validation of each run, not its best epoch.
    Smoke runs are explicitly labelled; absent metrics are never filled with zero.
    """
    priority = 'LOWEST'

    def before_run(self, runner):
        self.history = []
        try:
            revision = subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = bool(subprocess.check_output(
                ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = 'unknown', True
        settings = runner.cfg.get('reliability_experiment', {})
        self.record = dict(run_id=str(Path(runner.work_dir).resolve()) + '/' + runner.timestamp,
                           variant=settings.get('variant', 'reliability-gate'),
                           purpose=settings.get('purpose', 'finetune'),
                           work_dir=os.path.relpath(runner.work_dir, ROOT),
                           git_commit=revision, git_dirty_at_start=dirty,
                           config=runner.cfg.filename,
                           initialization_checkpoint=runner.cfg.get('load_from'),
                           random_seed=runner.seed,
                           status='started', training_completed=False)

    def _save(self, runner, phase, metrics=None):
        if not is_main_process():
            return
        entry = self.record
        entry['updated_at'] = datetime.now().astimezone().isoformat()
        entry['phase'] = phase
        entry['status'] = ('smoke_only' if entry['purpose'] == 'smoke'
                           else phase)
        if metrics is not None:
            numbers = {str(key): float(value) for key, value in metrics.items()}
            for name, suffix in [('eaa_map', 'entire_area/map'), ('dc_map', 'roi/map')]:
                entry[name] = next((v for k, v in numbers.items() if k.endswith(suffix)), None)
            loop = runner.val_loop if phase == 'validation' else runner.test_loop
            entry['samples'] = len(loop.dataloader.dataset)
            if phase == 'validation':
                entry['validation_epoch'] = runner.epoch
            self.history.append(dict(phase=phase, epoch=entry.get('validation_epoch'),
                                     metrics=numbers, time=entry['updated_at']))
        if phase == 'training_completed':
            entry['training_completed'] = True
            if runner._train_loop is not None:
                entry['iterations'] = runner.iter
                entry['train_samples'] = len(runner.train_loop.dataloader.dataset)
            try:
                entry['last_logged_train_loss'] = float(
                    runner.message_hub.get_scalar('train/loss').current())
            except KeyError:
                pass
        checkpoint = runner.message_hub.get_info('last_ckpt')
        if checkpoint:
            entry['last_checkpoint'] = checkpoint
        _atomic_write(Path(runner.work_dir) / 'reliability_result.json',
                      json.dumps(dict(summary=entry, history=self.history),
                                 ensure_ascii=False, indent=2) + '\n')
        try:
            update_result_files(ROOT, entry)
        except (OSError, ValueError) as error:
            runner.logger.warning('Could not update experiment README: %s. '
                                  'Metrics remain in reliability_result.json.', error)

    def after_val_epoch(self, runner, metrics=None):
        self._save(runner, 'validation', metrics)

    def after_test_epoch(self, runner, metrics=None):
        self._save(runner, 'test_only', metrics)

    def after_train(self, runner):
        self._save(runner, 'training_completed')
