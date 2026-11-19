"""Summarize V3 fixed-weight tests and paired heavy-occlusion deltas."""
import json
from pathlib import Path
import statistics
import sys


def field(metrics, suffix):
    value, = [x for k, x in metrics.items() if k.endswith('/' + suffix)]
    return value


def original_record(seed, arm):
    root = Path('work_dirs/reliability_fusion_v3')
    if arm == 'dense':
        pointer = 'last_paired_path.txt' if seed == 0 else 'last_dense_control_repeats_path.txt'
        parent = Path((root / pointer).read_text().strip())
        run = parent / ('dense_control' if seed == 0 else f'seed{seed}')
    else:
        prior = Path((root / 'last_prior_repeats_path.txt').read_text().strip())
        pairs = json.loads((prior / 'comparison.json').read_text())['pairs']
        pair, = [p for p in pairs if p['seed'] == seed]
        run = Path(pair['arms'][arm]['work_dir'])
    return json.loads((run / 'reliability_result.json').read_text())


def mean_sd(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))


def main():
    work = Path(sys.argv[1])
    raw = json.loads((work / 'accuracy/summary.json').read_text())
    assert raw['protocol']['frames'] == 1296
    rows = {(r['seed'], r['arm'], r['epoch'], r['condition']): r for r in raw['rows']}
    scores = {}
    for seed in range(3):
        scores[seed] = {}
        for arm in ('dense', 'guided'):
            row = rows[seed, arm, 6, 'center_42']
            masked = row['metrics']
            logged = original_record(seed, arm)
            clean = (rows[seed, arm, 6, 'clean']['metrics'] if seed == 2 else
                     next(x['metrics'] for x in logged['history']
                          if x['phase'] == 'validation' and x['epoch'] == 6))
            selected = {'eaa_map': 'entire_area/map', 'dc_map': 'roi/map',
                        'pedestrian_eaa_ap': 'entire_area/Pedestrian_3d_all',
                        'pedestrian_dc_ap': 'roi/Pedestrian_3d_all',
                        'pedestrian_eaa_r40': 'entire_area/Pedestrian_3d_all_r40',
                        'pedestrian_dc_r40': 'roi/Pedestrian_3d_all_r40'}
            values = {k: field(masked, s) for k, s in selected.items()}
            for k in ('recall_pct', 'fp_per_frame', 'tp', 'fp', 'fn'):
                values['pedestrian_eaa_' + k] = row['pedestrian_eaa'][k]
            values.update({k + '_masked_minus_clean': field(masked, s) - field(clean, s)
                           for k, s in selected.items()})
            scores[seed][arm] = values
        a, d = scores[seed]['dense'], scores[seed]['guided']
        scores[seed]['guided_minus_dense'] = {k: d[k] - a[k] for k in a}

    comparisons = {arm: {k: mean_sd([scores[seed][arm][k] for seed in range(3)])
                        for k in scores[0][arm]}
                   for arm in ('dense', 'guided', 'guided_minus_dense')}
    cyclist = {}
    for arm, epoch in [('dense', 5), ('dense', 6), ('guided', 6)]:
        row = rows[2, arm, epoch, 'clean']
        previous = original_record(2, arm)
        original = next(h['metrics'] for h in previous['history']
                        if h['phase'] == 'validation' and h['epoch'] == epoch)
        now = row['cyclist_dc_curve']
        cyclist[f'{arm}_epoch{epoch}'] = dict(
            ap11=now['ap11'], ap40=now['ap40'],
            original_ap11=field(original, 'roi/Cyclist_3d_all'),
            original_ap40=field(original, 'roi/Cyclist_3d_all_r40'),
            ap11_precisions=[now['precision'][i] for i in now['ap11_slot_indices']],
            ap11_recalls=[now['recall'][i] for i in now['ap11_slot_indices']],
            max_recall=max(now['recall']),
            operating_points=now['operating_points'])
    output = dict(protocol=raw['protocol'], cyclist_dc_seed2=cyclist,
                  center_42=dict(seeds=scores, statistics=comparisons,
                                 clean_reference='fixed-weight repeat for seed 2; final training validation for seed 0/1'))
    path = work / 'analysis.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print('ANALYSIS_COMPLETED', path, flush=True)


if __name__ == '__main__':
    main()
