"""Evaluate control_aug with B filtering and compare with archived V2-B runs."""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine import load
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmdet3d.evaluation.functional.kitti_utils.eval_vod import calculate_iou_partly, clean_data, compute_statistics_jit
from mmdet3d.evaluation.metrics.vod_metric import VODMetric
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

sys.path.insert(0,str(Path.cwd()/'tools/analysis_tools'))
from reliability_robustness import align_sample_indices, corrupt_batch, file_hash, SETTINGS

base=Path('work_dirs/reliability_fusion_v2')
nas=base/'ablation/nas'
out=Path(sys.argv[1])
ann=Path('data/vod/kitti_infos_val.pkl')
conditions=('clean','blur','occlusion')
training=json.loads(Path('repro/reliability_fusion_v2/multiseed/training-seeds0-1-2-20261031.json').read_text())
annotations=load(ann)['data_list']
indices=list(range(len(annotations)))
assert len(indices)==1296
register_all_modules()
MMLogger.get_instance('control_b',log_file=str(out/'metrics.log'))
results={}

def source(seed):
    return (nas/'head_filter_2x2_20261103_154357/full/AB' if seed==0
            else nas/f'head_filter_ab_repeats_20261103_161614/seed{seed}')

def entry(seed,name):
    return next(r for r in training if r['training_seed']==seed and r['variant']==name)

for seed in range(3):
    target=out/f'seed{seed}'
    target.mkdir()
    ref=source(seed)
    v2=json.loads((ref/'summary.json').read_text())
    assert v2['valid'] and v2['indices']==indices and v2['corruption_settings']==SETTINGS
    assert v2['annotation_sha256']==file_hash(ann)
    assert v2['checkpoints']['gate']['sha256']==entry(seed,'gate_main')['checkpoint_sha256']
    b=Config.fromfile(ref/'gate_resolved.py')
    assert b.model.test_cfg.nms_pre==300 and b.model.test_cfg.score_thr==.1
    cfg=Config.fromfile('configs/experiments/reliability_fusion_v2/control_aug.py')
    cfg.model.test_cfg.nms_pre=300
    cfg.model.test_cfg.score_thr=.1
    cfg.test_dataloader.dataset.indices=indices
    cfg.test_dataloader.num_workers=0
    cfg.test_dataloader.persistent_workers=False
    cfg.dump(str(target/'resolved.py'))
    rec=entry(seed,'control_aug')
    checkpoint=Path(rec['summary']['last_checkpoint'])
    assert file_hash(checkpoint)==rec['checkpoint_sha256']
    torch.manual_seed(0)
    model=MODELS.build(cfg.model).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
    loader=Runner.build_dataloader(cfg.test_dataloader,seed=0)
    results[seed]=dict(checkpoint_sha256=rec['checkpoint_sha256'],v2_source=str(ref),conditions={})
    for c in conditions:
        print('START',seed,c,flush=True)
        torch.manual_seed(0)
        np.random.seed(0)
        directory=target/c
        directory.mkdir()
        metric=copy.deepcopy(cfg.test_evaluator)
        metric[0]['pklfile_prefix']=str(directory/'predictions')
        evaluator=Evaluator(metric)
        evaluator.dataset_meta=loader.dataset.metainfo
        offset=0
        with torch.no_grad():
            for data in loader:
                data=corrupt_batch(data,c)
                pred=model.test_step(data)
                offset=align_sample_indices(pred,indices,offset,annotations)
                evaluator.process(data_samples=pred,data_batch=data)
                if offset%128==0:
                    print('PROGRESS',seed,c,offset,len(indices),flush=True)
        assert offset==len(indices)
        ctrl=evaluator.evaluate(offset)
        prev=next(r['metrics'] for r in v2['results'] if r['model']=='gate' and r['condition']==c)
        def select(r):
            keys={'eaa':'entire_area/map','dc':'roi/map','pedestrian_ap':'entire_area/Pedestrian_3d_all'}
            return {k:float(next(v for n,v in r.items() if n.endswith(s))) for k,s in keys.items()}
        results[seed]['conditions'][c]=dict(control=select(ctrl),v2=select(prev))
        print('RESULT',seed,c,results[seed]['conditions'][c],flush=True)
        (out/'progress.json').write_text(json.dumps(results,indent=2))
    assert file_hash(checkpoint)==rec['checkpoint_sha256']
    del model
    torch.cuda.empty_cache()

gt=[r['kitti_annos'] for r in VODMetric(ann_file=str(ann)).convert_annos_to_kitti_annos(load(ann))]
def table(a,dt=False):
    return np.column_stack([np.asarray(a['bbox']).reshape(-1,4),a['alpha']]+([a['score']] if dt else [])).astype('float64')
for seed, item in results.items():
    for c,row in item['conditions'].items():
        for name, directory in [('control',out/f'seed{seed}'/c),('v2',source(seed)/'gate'/c)]:
            with (directory/'predictions/pred_instances_3d.pkl').open('rb') as f:
                dt=pickle.load(f)
            assert len(dt)==len(gt) and all(np.all(a['sample_idx']==i) for i,a in enumerate(dt))
            overlaps=calculate_iou_partly(dt,gt,2,num_parts=130)[0]
            total=np.zeros(3,int)
            for o,g,d in zip(overlaps,gt,dt):
                _,ig,idd,dc=clean_data(g,d,1,0)
                total+=compute_statistics_jit(o,table(g),table(d,True),np.asarray(ig,dtype='int64'),
                    np.asarray(idd,dtype='int64'),np.asarray(dc,dtype='float64').reshape(-1,4),2,.25,.13,True,False)[:3]
            tp,fp,fn=map(int,total)
            row[name].update(ped_tp=tp,ped_fp=fp,ped_recall=tp/(tp+fn),ped_fp_frame=fp/len(gt))
        print('OPERATING_POINT',seed,c,row,flush=True)
(out/'comparison.json').write_text(json.dumps(dict(valid=True,frames=len(gt),settings=SETTINGS,
    nms_pre=300,score_thr=.10,pedestrian_output_cutoff=.13,seeds=results),indent=2))
print('ALL_COMPLETED',out/'comparison.json',flush=True)
