"""Run the three matched V2 experiments sequentially on one GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('control_clean', 'control_aug', 'gate_main')


def now():
    return datetime.now().astimezone().isoformat()


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def source_hashes():
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '*.py', '*.sh'], text=True).split('\0')
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if p}


def run(output):
    lock = (ROOT / 'work_dirs/reliability_fusion_v2/.gpu0-queue.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Commit working-tree changes before starting this queue')
    output.mkdir(parents=True, exist_ok=False)
    code = revision()
    hashes = source_hashes()
    state = dict(started_at=now(), status='running', pid=os.getpid(),
                 git_commit=code, source_sha256=hashes, variants=list(VARIANTS), runs=[])

    def save():
        temp = output / 'status.json.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temp.replace(output / 'status.json')

    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0',
               NUMBA_CUDA_USE_NVIDIA_BINDING='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        for variant in VARIANTS:
            if revision() != code or source_hashes() != hashes:
                raise RuntimeError('Code changed since queue start; stopping before next experiment')
            work_dir = ROOT / 'work_dirs/reliability_fusion_v2' / variant / output.name
            work_dir.mkdir(parents=True, exist_ok=False)
            config = f'configs/experiments/reliability_fusion_v2/{variant}.py'
            command = [sys.executable, '-u', 'tools/train.py', config,
                       '--work-dir', str(work_dir)]
            entry = dict(variant=variant, started_at=now(), status='running',
                         work_dir=str(work_dir), command=command)
            state['runs'].append(entry)
            state['current_variant'] = variant
            save()
            print(now(), 'START', variant, work_dir, flush=True)
            with (work_dir / 'console.log').open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                entry['pid'] = child.pid
                save()
                entry['returncode'] = child.wait()
            entry['finished_at'] = now()
            if entry['returncode']:
                entry['status'] = 'failed'
                raise RuntimeError(f'{variant} exited with code {entry["returncode"]}')
            result = json.loads((work_dir / 'reliability_result.json').read_text())
            summary = result['summary']
            if not summary.get('training_completed') or summary.get('validation_epoch') != 6:
                raise RuntimeError(f'{variant} did not finish six epochs with final validation')
            if not (work_dir / 'epoch_6.pth').is_file():
                raise RuntimeError(f'{variant} has no final epoch_6.pth')
            entry.update(status='completed', eaa_map=summary['eaa_map'],
                         dc_map=summary['dc_map'])
            save()
            print(now(), 'DONE', variant, entry['eaa_map'], entry['dc_map'], flush=True)
        state.update(status='completed', finished_at=now(), current_variant=None)
        save()
        print(now(), 'ALL_COMPLETED', flush=True)
    except Exception as error:
        state.update(status='failed', finished_at=now(), error=str(error))
        if state['runs'] and state['runs'][-1]['status'] == 'running':
            state['runs'][-1]['status'] = 'failed'
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, help='New directory for queue status')
    args = parser.parse_args()
    os.chdir(ROOT)
    directory = (ROOT / args.work_dir).resolve()
    if (ROOT / 'work_dirs/reliability_fusion_v2').resolve() not in directory.parents:
        raise ValueError('Queue output must be inside work_dirs/reliability_fusion_v2')
    run(directory)
