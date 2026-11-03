"""Run paired seed 1/2 A/B filter evaluation and measure test_step latency."""
import argparse
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
TOOLS = ROOT / 'tools/analysis_tools'
sys.path.insert(0, str(TOOLS))
ROBUSTNESS = TOOLS / 'reliability_robustness.py'
WORK = Path('work_dirs/reliability_fusion_v2')
SOURCE = WORK / 'ablation/nas/head_filter_2x2_20261103_154357'
ANN = Path('data/vod/kitti_infos_val.pkl')
CONDITIONS = ('clean', 'blur', 'occlusion')
N = 1296
WARMUP = 30


def timed_eval(out, weights):
    import torch
    from mmdet3d.models.detectors.voxelnet_fuse import VoxelNetFuse

    original = VoxelNetFuse.test_step
    times = []

    def timed(self, data):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(self, data)
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
        return result

    VoxelNetFuse.test_step = timed
    sys.argv = [str(ROBUSTNESS), '--samples', str(N), '--conditions', *CONDITIONS,
                '--control-checkpoint', str(weights), '--gate-checkpoint', str(weights),
                '--control-config', str(SOURCE/'A.py'), '--gate-config', str(SOURCE/'B.py'),
                '--work-dir', str(out), '--no-record-readme']
    try:
        runpy.run_path(str(ROBUSTNESS), run_name='__main__')
    finally:
        VoxelNetFuse.test_step = original
    assert len(times) == 2 * N * len(CONDITIONS)
    groups = {}
    for i, (policy, condition) in enumerate((p,c) for p in ('A','B') for c in CONDITIONS):
        kept = times[i*N+WARMUP:(i+1)*N]
        groups.setdefault(policy, {})[condition] = dict(
            median_ms=float(np.median(kept)), p95_ms=float(np.percentile(kept,95)),
            measured_frames=len(kept), warmup_frames=WARMUP)
    (out/'latency.json').write_text(json.dumps(dict(
        scope='CUDA-synchronized test_step: preprocessing, model forward, decode and NMS; excludes loading, corruption and evaluation',
        groups=groups, raw_ms=times), indent=2))


def summarize(out):
    import pickle
    from mmdet3d.evaluation.metrics.vod_metric import VODMetric
    from mmdet3d.evaluation.functional.kitti_utils.eval_vod import (
        calculate_iou_partly, clean_data, compute_statistics_jit)
    from reliability_robustness import file_hash

    def read(path):
        with path.open('rb') as stream:
            return pickle.load(stream)

    def data(a, det=False):
        return np.column_stack([np.asarray(a['bbox']).reshape(-1,4), a['alpha']]
                               + ([a['score']] if det else [])).astype('float64')

    gt = [x['kitti_annos'] for x in VODMetric(ann_file=str(ANN)).convert_annos_to_kitti_annos(read(ANN))]
    result = {}
    for seed in (1,2):
        run = out/f'seed{seed}'
        r = json.loads((run/'summary.json').read_text())
        latency = json.loads((run/'latency.json').read_text())['groups']
        assert r['valid'] and r['indices']==list(range(N))
        result[seed] = {}
        for row in r['results']:
            tag, c = row['model'], row['condition']
            label, threshold = ('A',.10) if tag=='control' else ('B',.13)
            record = dict(eaa_map=row['eaa_map'],dc_map=row['dc_map'],
                          pedestrian_ap=next(v for k,v in row['metrics'].items() if k.endswith('entire_area/Pedestrian_3d_all')),
                          latency=latency[label][c])
            path = run/tag/c/'predictions/pred_instances_3d.pkl'
            dt = read(path)
            assert len(dt)==len(gt)==N and all(np.all(d['sample_idx']==i) for i,d in enumerate(dt))
            ov = calculate_iou_partly(dt,gt,2,num_parts=130)[0]
            counts = np.zeros(3,int)
            for i,(g,d) in enumerate(zip(gt,dt)):
                _,ig,idd,dc = clean_data(g,d,1,0)
                counts += compute_statistics_jit(ov[i],data(g),data(d,True),
                    np.asarray(ig,dtype='int64'),np.asarray(idd,dtype='int64'),
                    np.asarray(dc,dtype='float64').reshape(-1,4),2,.25,threshold,True,False)[:3]
            tp,fp,fn = map(int,counts)
            record['pedestrian_operating_point'] = dict(score_cutoff=threshold,tp=tp,fp=fp,fn=fn,
                fp_per_frame=fp/N,recall=tp/(tp+fn))
            record['predictions_sha256'] = file_hash(path)
            result[seed].setdefault(label,{})[c]=record
            print('COMPARE',seed,label,c,record,flush=True)
    (out/'comparison.json').write_text(json.dumps(dict(valid=True,frames=N,
         operating_point='Fixed post-output pedestrian cutoff A=.10 B=.13; AP/EAA/DC use native detector output',
         seeds=result), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--worker-checkpoint', type=Path)
    args = parser.parse_args()
    out = args.work_dir
    if args.worker_checkpoint:
        timed_eval(out,args.worker_checkpoint)
        return
    from mmengine.config import Config
    from reliability_robustness import file_hash
    for label, topk in [('A',100),('B',300)]:
        cfg = Config.fromfile(SOURCE/f'{label}.py')
        assert cfg.model.test_cfg.nms_pre==topk and cfg.model.test_cfg.score_thr==.1
    train = json.loads(Path('repro/reliability_fusion_v2/multiseed/training-seeds0-1-2-20261031.json').read_text())
    for seed in (1,2):
        entry = next(r for r in train if r['training_seed']==seed and r['variant']=='gate_main')
        weights = Path(entry['summary']['last_checkpoint'])
        assert file_hash(weights)==entry['checkpoint_sha256']
        run = out/f'seed{seed}'
        run.mkdir(exist_ok=False)
        print('START',seed,flush=True)
        with (run/'console.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--work-dir',str(run),
                            '--worker-checkpoint',str(weights)],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert file_hash(weights)==entry['checkpoint_sha256']
        print('DONE',seed,flush=True)
    summarize(out)
    print('ALL_COMPLETED',out/'comparison.json',flush=True)


if __name__=='__main__':
    main()
