"""Repeat learned vs backbone-query identity on existing full-V2 seeds 1 and 2."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.work_dir).resolve()
    training = json.loads((ROOT / 'repro/reliability_fusion_v2/multiseed/training-seeds0-1-2-20261031.json').read_text())
    sources = git('ls-files', '-z', '--', '*.py', '*.sh').split('\0')
    source_hashes = {name: digest(name) for name in sources if name}
    code = git('rev-parse', 'HEAD')
    jobs = []
    for seed in (1, 2):
        record = next(row for row in training if row['training_seed'] == seed and row['variant'] == 'gate_main')
        checkpoint = Path(record['summary']['last_checkpoint']).resolve()
        expected_hash = record['checkpoint_sha256']
        if digest(checkpoint) != expected_hash:
            raise RuntimeError(f'Checkpoint differs from the original training archive: {checkpoint}')
        directory = output / f'seed{seed}'
        command = [sys.executable, '-u', str(ROOT / 'tools/analysis_tools/reliability_attention_intervention.py'),
                   '--gate-kind', 'query', '--policies', 'learned', 'backbone_identity',
                   '--training-seed', str(seed), '--checkpoint', str(checkpoint),
                   '--samples', '1296', '--conditions', 'clean', 'dark', 'blur', 'occlusion',
                   '--work-dir', str(directory), '--record-readme']
        jobs.append(dict(training_seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=expected_hash,
                         output_dir=str(directory), command=command))
    if args.dry_run:
        for job in jobs:
            print('VERIFIED_CHECKPOINT', job['training_seed'], job['checkpoint_sha256'])
            print(shlex.join(job['command']))
        print('DRY_RUN_OK: no evaluation, training or output files created')
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'batch.json').exists() or any((output / f'seed{seed}').exists() for seed in (1, 2)):
        raise FileExistsError('Use a fresh batch directory to preserve previous results')
    report = dict(started_at=datetime.now().astimezone().isoformat(), training=False,
                  valid=False, status='running', git_commit=code, source_sha256=source_hashes,
                  jobs=jobs, completed=[], policies=['learned', 'backbone_identity'],
                  samples=1296, conditions=['clean', 'dark', 'blur', 'occlusion'],
                  evaluation_seed=0, purpose='Replicate the seed-0 backbone-query intervention; not a new trained model')
    save(output / 'batch.json', report)
    try:
        for job in jobs:
            if git('rev-parse', 'HEAD') != code or any(digest(name) != value for name, value in source_hashes.items()):
                raise RuntimeError('Code changed since this batch started; do not mix implementations')
            if digest(job['checkpoint']) != job['checkpoint_sha256']:
                raise RuntimeError('Source checkpoint changed')
            directory = Path(job['output_dir'])
            directory.mkdir(parents=True, exist_ok=True)
            print('START', job['training_seed'], directory, flush=True)
            environment = dict(os.environ, NUMBA_CUDA_USE_NVIDIA_BINDING='1', PYTHONDONTWRITEBYTECODE='1')
            environment.setdefault('CUDA_VISIBLE_DEVICES', '0')
            with (directory / 'console.log').open('w') as stream:
                subprocess.run(job['command'], env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            result_path = directory / 'summary.json'
            result = json.loads(result_path.read_text())
            if not (result['valid'] and result['training_seed'] == job['training_seed']
                    and result['checkpoint_unchanged'] and result['samples'] == 1296
                    and result['checkpoint']['sha256'] == job['checkpoint_sha256']
                    and result['policies'] == ['learned', 'backbone_identity']
                    and len(result['results']) == 8):
                raise RuntimeError('Completed child result does not match the batch protocol')
            report['completed'].append(dict(training_seed=job['training_seed'], summary=str(result_path),
                                            sha256=digest(result_path), run_id=result['run_id']))
            save(output / 'batch.json', report)
            print('DONE', job['training_seed'], flush=True)
        report.update(valid=True, status='completed', completed_at=datetime.now().astimezone().isoformat())
        save(output / 'batch.json', report)
        print('ALL_COMPLETED', output / 'batch.json', flush=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error), updated_at=datetime.now().astimezone().isoformat())
        save(output / 'batch.json', report)
        raise


if __name__ == '__main__':
    main()
