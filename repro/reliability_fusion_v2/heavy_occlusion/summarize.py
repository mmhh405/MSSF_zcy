import json
import pickle
import statistics
import sys
from pathlib import Path
import numpy as np
from mmdet3d.evaluation.metrics.vod_metric import VODMetric
from mmdet3d.evaluation.functional.kitti_utils.eval_vod import calculate_iou_partly,clean_data,compute_statistics_jit
b=Path(sys.argv[1])
s=json.loads((b/'full/summary.json').read_text())
assert s['valid'] and s['indices']==list(range(1296))
ann=Path('data/vod/kitti_infos_val.pkl')
gt=[x['kitti_annos'] for x in VODMetric(ann_file=str(ann)).convert_annos_to_kitti_annos(pickle.loads(ann.read_bytes()))]
def tab(a,det=False):
 return np.column_stack([np.asarray(a['bbox']).reshape(-1,4),a['alpha']]+([a['score']] if det else [])).astype('float64')
all_results={}
for c in s['mask_xyxy_fraction']:
 values={}
 for seed,pair in s['seeds'].items():
  a={}
  for model in ['control','v2']:
   f=b/'full'/f'seed{seed}'/model/c/'predictions/pred_instances_3d.pkl'
   dt=pickle.loads(f.read_bytes())
   assert len(dt)==1296 and all(np.all(d['sample_idx']==i) for i,d in enumerate(dt))
   ov=calculate_iou_partly(dt,gt,2,num_parts=130)[0]
   cnt=np.zeros(3,int)
   for o,g,d in zip(ov,gt,dt):
    _,ig,idd,dc=clean_data(g,d,1,0)
    cnt+=compute_statistics_jit(o,tab(g),tab(d,True),np.asarray(ig,dtype='int64'),np.asarray(idd,dtype='int64'),np.asarray(dc,dtype='float64').reshape(-1,4),2,.25,.13,True,False)[:3]
   t,f,n=map(int,cnt)
   row=pair[model]['conditions'][c]
   a[model]=dict(**{k:row[k] for k in ['eaa','dc','pedestrian_ap']},tp=t,fp=f,ped_recall_pct=100*t/(t+n),ped_fp_per_frame=f/1296)
  a['delta']={k:a['v2'][k]-v for k,v in a['control'].items()}
  values[seed]=a
  print(c,seed,{k:round(v,5) for k,v in a['delta'].items()},flush=True)
 all_results[c]=dict(seeds=values,mean={k:statistics.mean(v['delta'][k] for v in values.values()) for k in a['delta']},sample_sd={k:statistics.stdev(v['delta'][k] for v in values.values()) for k in a['delta']})
 print('MEAN',c,all_results[c]['mean'],'SD',all_results[c]['sample_sd'],flush=True)
(b/'comparison.json').write_text(json.dumps(dict(samples=1296,pedestrian_cutoff=.13,mask_xyxy_fraction=s['mask_xyxy_fraction'],conditions=all_results),indent=2))
print('SUMMARY_COMPLETE',b/'comparison.json',flush=True)
