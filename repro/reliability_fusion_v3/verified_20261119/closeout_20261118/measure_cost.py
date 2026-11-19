"""Measure the trained dense, independent, and guided models when the GPU is idle."""
import fcntl
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from evaluate import evaluations, save

SAMPLES = 128
WARMUP = 10
DESKTOP = ('gnome-remote-desktop', 'gnome-shell', '/Xorg')


def clients(own_pid=None):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name',
         '--format=csv,noheader,nounits'], text=True)
    return [line for line in output.splitlines()
            if line.split(',')[0].strip() != str(own_pid)
            and not any(name in line for name in DESKTOP)]


def wait_for_profile_slot(lock):
    reported = False
    while True:
        other = clients()
        if not other:
            time.sleep(4)
            if not clients():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if not clients():
                        return
                    fcntl.flock(lock, fcntl.LOCK_UN)
        elif not reported:
            print('WAITING_FOR_IDLE_GPU', other, flush=True)
            reported = True
        time.sleep(20)


def profile_one(run, mode, out):
    config, = run.glob('*.py')
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'profile.json'
    command = [sys.executable, '-u', 'tools/analysis_tools/profile_reliability_v3.py',
               '--config', str(config), '--checkpoint', str(run / 'epoch_6.pth'),
               '--samples', str(SAMPLES), '--warmup', str(WARMUP),
               '--modes', mode, '--output', str(path)]
    competing = set()
    with (out / 'console.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            competing.update(clients(process.pid))
            time.sleep(1)
        code = process.returncode
    observed = json.loads(path.read_text()) if path.is_file() else None
    valid = (code == 0 and observed is not None and
             observed['performance_comparison_valid'] and not competing and
             not observed['uninitialized_router'])
    save(out / 'audit.json', dict(valid=valid, competing_clients=sorted(competing),
                                  returncode=code))
    if not valid:
        log_text = (out / 'console.log').read_text(errors='replace')
        if competing or 'GPU has other compute clients' in log_text or (
                observed and observed['competing_compute_clients_at_end']):
            return None
        raise RuntimeError(f'Profiler failed. See {out / "console.log"}')
    sample = observed['modes'][mode]
    rows = sample['rows']
    assert len(rows) == SAMPLES
    latencies = sorted(x['wall_ms'] for x in rows)
    result = dict(median_ms=sample['median_wall_ms'],
                  p95_ms=latencies[math.ceil(.95 * len(latencies)) - 1],
                  peak_allocated_mib=sample['max_peak_allocated_bytes'] / 1048576,
                  parameter_bytes=observed['parameter_bytes'],
                  checkpoint_sha256=observed['checkpoint_sha256'],
                  raw=str(path))
    if mode == 'select':
        result.update(fpn_feature_mib=statistics.mean(
                          x['fpn_persistent_bytes'] for x in rows) / 1048576,
                      matched_dense_fpn_reference_mib=statistics.mean(
                          x['fpn_dense_reference_bytes'] for x in rows) / 1048576,
                      image_pixel_work_ratio=statistics.mean(
                          x['image_pixel_work_ratio'] for x in rows))
    return result


def main():
    folder = Path(sys.argv[1]).resolve()
    assert (folder / 'accuracy/summary.json').is_file()
    runs = evaluations()
    lock = (folder.parent / '.paired-training.lock').open('a')
    trials = []
    arms = ('dense', 'independent', 'guided')
    for seed in sorted(runs):
        order = arms[seed:] + arms[:seed]
        for repeat, ordered in enumerate((order, order[::-1]), start=1):
            attempt = 0
            while True:
                wait_for_profile_slot(lock)
                attempt += 1
                print('START_RESOURCE', seed, repeat, list(ordered), attempt, flush=True)
                values = {}
                for arm in ordered:
                    target = folder / 'resources' / f'seed{seed}' / f'rep{repeat}_attempt{attempt}' / arm
                    result = profile_one(runs[seed][arm], 'dense' if arm == 'dense' else 'select', target)
                    if result is None:
                        print('RETRY_RESOURCE', seed, repeat, arm, flush=True)
                        break
                    values[arm] = result
                fcntl.flock(lock, fcntl.LOCK_UN)
                if len(values) == len(arms):
                    trials.append(dict(seed=seed, repeat=repeat, order=ordered, arms=values))
                    save(folder / 'resources/progress.json', dict(trials=trials))
                    print('DONE_RESOURCE', seed, repeat, flush=True)
                    break
    comparisons = []
    for row in trials:
        base = row['arms']['dense']
        for arm in ('independent', 'guided'):
            tested = row['arms'][arm]
            comparisons.append(dict(seed=row['seed'], repeat=row['repeat'], arm=arm,
                median_latency_reduction_pct=100*(1-tested['median_ms']/base['median_ms']),
                p95_latency_reduction_pct=100*(1-tested['p95_ms']/base['p95_ms']),
                peak_allocated_reduction_pct=100*(1-tested['peak_allocated_mib']/base['peak_allocated_mib']),
                fpn_feature_reduction_pct=100*(1-tested['fpn_feature_mib'] /
                                                   tested['matched_dense_fpn_reference_mib'])))
    save(folder / 'resources/summary.json', dict(trials=trials, comparisons=comparisons,
         timed_frames=SAMPLES, warmup_frames=WARMUP, threads=4, batch_size=1,
         fpn_baseline='shape-calculated dense FPN bytes for each routed frame',
         latency_scope='GPU synchronized test_step; excludes dataset loading'))
    print('ALL_COMPLETED', folder / 'resources/summary.json', flush=True)


if __name__ == '__main__':
    main()
