"""Paired seeds 1/2 for equal-budget V3 priors; resume interrupted runs.

Launch with the project environment, from any directory. Fixed endpoint: epoch 6.
The independent arm disables early cross-modal priors, keeping V2 fusion gates.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    'guided': 'configs/experiments/reliability_fusion_v3/gate_image_budget_only.py',
    'independent': 'configs/experiments/reliability_fusion_v3/gate_image_budget_independent.py',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def checkpoint_epoch(path):
    """Reject truncated, nonfinite, or non-resumable training checkpoints."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    epoch = checkpoint.get('meta', {}).get('epoch')
    require(isinstance(epoch, int) and 1 <= epoch <= 6, 'Invalid checkpoint epoch')
    require(checkpoint['meta'].get('iter') == epoch * 5139, 'Unexpected checkpoint iteration')
    require(bool(checkpoint.get('state_dict')), 'Missing model state')
    require('optimizer' in checkpoint and 'param_schedulers' in checkpoint,
            'Missing optimizer or scheduler state')
    require(all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values()
                if v.is_floating_point()), 'Nonfinite model state')
    return epoch


def completed_result(run, seed):
    path = run / 'reliability_result.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    s = data['summary']
    if not s.get('training_completed'):
        return None
    expected = dict(random_seed=seed, validation_epoch=6, iterations=30834,
                    train_samples=5139, samples=1296)
    require(all(s.get(k) == v for k, v in expected.items()), 'Completed-run protocol mismatch: ' + str(run))
    require(all(isinstance(s.get(k), (int, float)) and math.isfinite(s[k])
                for k in ('eaa_map', 'dc_map')), 'Missing finite final metrics')
    require(checkpoint_epoch(run / 'epoch_6.pth') == 6, 'Invalid final checkpoint')
    return s


def select_resume(run):
    valid = []
    for path in sorted(run.glob('epoch_*.pth')):
        try:
            valid.append((checkpoint_epoch(path), path))
        except Exception as error:
            # Preserve bad files so a later training save cannot overwrite them.
            recovery = run / 'rejected_checkpoints'
            recovery.mkdir(exist_ok=True)
            target = recovery / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + path.name)
            path.rename(target)
            print('REJECTED_CHECKPOINT', path, str(error), flush=True)
    return max(valid, default=(0, None), key=lambda x: x[0])


def preflight():
    import torch
    import mmengine
    from mmengine.config import Config
    require(torch.cuda.is_available(), 'CUDA unavailable in ' + sys.executable)
    configs = [Config.fromfile(str(ROOT / p)) for p in CONFIGS.values()]
    models = [copy.deepcopy(c.model) for c in configs]
    require(models[0]['early_budget']['router'].get('use_radar_to_image', True)
            and models[0]['early_budget']['router'].get('use_image_to_radar', True),
            'Guided config must enable both prior switches')
    require(models[1]['early_budget']['router'].get('use_radar_to_image') is False
            and models[1]['early_budget']['router'].get('use_image_to_radar') is False,
            'Independent config must disable both prior switches')
    for model in models:
        for key in ('use_radar_to_image', 'use_image_to_radar'):
            model['early_budget']['router'].pop(key, None)
    require(models[0] == models[1], 'Models differ beyond prior switches')
    for key in ('train_cfg', 'train_dataloader', 'val_dataloader', 'val_evaluator',
                'optim_wrapper', 'param_scheduler', 'randomness', 'load_from'):
        require(configs[0].get(key) == configs[1].get(key), 'Unmatched setting: ' + key)
    for cfg in configs:
        require(cfg.train_cfg.max_epochs == 6 and cfg.train_dataloader.batch_size == 1,
                'Expected six epochs, batch one')
        require(cfg.model.early_budget.router.radar_keep_ratio == 1., 'Radar budget must remain 100%')
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '*.py'], cwd=ROOT)
    paths = {p.decode() for p in tracked.split(b'\0') if p}
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    files = {p: sha256(ROOT / p) for p in sorted(paths)}
    for name in ('kitti_infos_train.pkl', 'kitti_infos_val.pkl'):
        p = 'data/vod/' + name
        files[p] = sha256(ROOT / p)
    checkpoint = (ROOT / configs[0].load_from).resolve()
    files[str(checkpoint)] = sha256(checkpoint)
    return dict(files=files, configs=CONFIGS, warm_start=str(checkpoint),
                python=sys.executable, torch=torch.__version__, mmengine=mmengine.__version__,
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)


def verify_files(manifest):
    for name, expected in manifest['protocol']['files'].items():
        require(sha256(ROOT / name) == expected, 'Input changed; do not mix experiment versions: ' + name)


def summarize(base, manifest):
    for path, expected in manifest['seed0_result_sha256'].items():
        require(sha256(path) == expected, 'Seed-0 evidence changed: ' + path)
    rows = []
    for seed, arms in [(0, manifest['seed0_paths'])] + [
            (seed, {arm: str(base / ('seed' + str(seed)) / arm) for arm in CONFIGS})
            for seed in manifest['seeds']]:
        pair = {arm: completed_result(Path(path), seed) for arm, path in arms.items()}
        require(all(pair.values()), 'Incomplete pair for seed ' + str(seed))
        rows.append(dict(seed=seed, arms=pair, delta_pp={
            key: pair['guided'][key] - pair['independent'][key] for key in ('eaa_map', 'dc_map')}))
    result = dict(metric='ordinary 3D AP; fixed epoch 6; guided minus independent',
                  conditional_on_shared_V2_seed0_initialization=True, pairs=rows, statistics={})
    for key in ('eaa_map', 'dc_map'):
        result['statistics'][key] = {
            arm: dict(mean=statistics.mean([r['arms'][arm][key] for r in rows]),
                      sample_sd=statistics.stdev([r['arms'][arm][key] for r in rows]))
            for arm in CONFIGS}
        deltas = [r['delta_pp'][key] for r in rows]
        result['statistics'][key]['paired_delta_pp'] = dict(
            mean=statistics.mean(deltas), sample_sd=statistics.stdev(deltas))
    write_json(base / 'comparison.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--check-only', action='store_true', help='Validate protocol without starting or creating runs')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(PYTHONPATH=str(ROOT), NUMBA_CUDA_USE_NVIDIA_BINDING='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ['PATH']
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    base = Path(args.work_dir).resolve()
    parent = ROOT / 'work_dirs/reliability_fusion_v3'
    require(parent.resolve() in base.parents, 'Work directory must be inside project work_dirs/reliability_fusion_v3')
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / '.paired-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = preflight()
        manifest_path = base / 'run_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            require(manifest['protocol'] == protocol, 'Protocol changed; restore sources/environment before resuming')
        else:
            seed0, seed0_hashes = {}, {}
            for arm, pointer in [('guided', 'last_image_budget_path.txt'),
                                 ('independent', 'last_image_budget_independent_path.txt')]:
                path = Path((parent / pointer).read_text().strip()).resolve()
                require(completed_result(path, 0) is not None, 'Seed 0 is incomplete: ' + str(path))
                seed0[arm] = str(path)
                seed0_hashes[str(path / 'reliability_result.json')] = sha256(path / 'reliability_result.json')
            manifest = dict(created_at=datetime.now().astimezone().isoformat(), protocol=protocol,
                            seeds=[1, 2], seed0_paths=seed0, seed0_result_sha256=seed0_hashes,
                            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                            git_status=subprocess.check_output(['git', 'status', '--short'], text=True))
        print('PREFLIGHT_OK', sys.executable, 'seeds=1,2 epochs=6 batch=1', flush=True)
        if args.check_only:
            return
        base.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            require(not any(base.glob('seed*')), 'Nonempty run directory without manifest')
            write_json(manifest_path, manifest)
        (parent / 'last_prior_repeats_path.txt').write_text(str(base) + '\n')
        (base / 'queue.pid').write_text(str(os.getpid()) + '\n')
        for seed in manifest['seeds']:
            order = ['guided', 'independent'] if seed == 1 else ['independent', 'guided']
            for arm in order:
                verify_files(manifest)
                run = base / ('seed' + str(seed)) / arm
                run.mkdir(parents=True, exist_ok=True)
                if completed_result(run, seed):
                    print('SKIP_COMPLETED', seed, arm, flush=True)
                    continue
                epoch, checkpoint = select_resume(run)
                require(epoch < 6, 'Final checkpoint exists without completed metrics; evaluate it before continuing: ' + str(run))
                attempt = run / 'attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                attempt.mkdir(parents=True)
                for name in ('reliability_result.json', 'last_checkpoint'):
                    if (run / name).is_file():
                        shutil.copy2(run / name, attempt / ('previous_' + name))
                console = run / 'console.log'
                if console.is_symlink():
                    console.unlink()
                elif console.exists():
                    console.rename(attempt / 'previous_console.log')
                console.symlink_to(attempt / 'console.log')
                command = [sys.executable, '-u', str(ROOT / 'tools/train.py'), str(ROOT / CONFIGS[arm]),
                           '--work-dir', str(run), '--cfg-options', 'randomness.seed=' + str(seed)]
                if checkpoint:
                    command.extend(['--resume', str(checkpoint)])
                status = dict(status='running', seed=seed, model=arm, work_dir=str(run),
                              resume_epoch=epoch, command=command)
                write_json(base / 'progress.json', status)
                print('START', seed, arm, 'resume_epoch=' + str(epoch), run, flush=True)
                with (attempt / 'console.log').open('w') as log:
                    code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    write_json(base / 'progress.json', dict(status, status='failed', returncode=code))
                    raise RuntimeError('Training failed; inspect ' + str(console))
                require(completed_result(run, seed) is not None, 'Training exited without final metrics: ' + str(run))
                print('DONE', seed, arm, flush=True)
        summarize(base, manifest)
        write_json(base / 'progress.json', dict(status='completed', comparison=str(base / 'comparison.json')))
        print('ALL_COMPLETED', base / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
