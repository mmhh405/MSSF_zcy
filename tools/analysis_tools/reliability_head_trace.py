"""Trace saved VoD pedestrian cases through the actual anchor-head filters."""
import argparse
import copy
import fcntl
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmengine import load
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.evaluation.functional.kitti_utils.eval_vod import d3_box_overlap
from mmdet3d.models.dense_heads import base_3d_dense_head as dense
from mmdet3d.models.layers.box3d_nms import nms_bev, nms_normal_bev
from mmdet3d.registry import METRICS, MODELS
from mmdet3d.structures import Box3DMode
from mmdet3d.utils import register_all_modules
from reliability_feature_drift import ANN, MODELS_TO_RUN, WORK
from reliability_robustness import ROOT, align_sample_indices, corrupt_batch, file_hash

THRESHOLD = .25


def all_candidates(head, inputs, cfg):
    boxes, scores, directions, selected = [], [], [], []
    offset = 0
    for cls, pred, direction, anchors in zip(*(inputs[k] for k in
            ('cls_score_list', 'bbox_pred_list', 'dir_cls_pred_list', 'mlvl_priors'))):
        score = cls.permute(1,2,0).reshape(-1, head.num_classes).sigmoid()
        pred = pred.permute(1,2,0).reshape(-1, head.box_code_size)
        boxes.append(head.bbox_coder.decode(anchors, pred))
        scores.append(score)
        directions.append(direction.permute(1,2,0).reshape(-1,2).argmax(-1))
        n = cfg.get('nms_pre', -1)
        indices = score.max(dim=1).values.topk(n).indices if 0 < n < len(score) else torch.arange(len(score),device=score.device)
        selected.append(indices + offset)
        offset += len(score)
    return tuple(torch.cat(x) for x in (boxes, scores, directions, selected))


def overlaps(boxes, target):
    if not len(boxes):
        return np.zeros(0)
    # A box with no overlap in its enclosing circle cannot pass 3D IoU.
    r = np.linalg.norm(boxes[:,[3,5]],axis=1)/2
    r0 = np.linalg.norm(target[[3,5]])/2
    indices = np.flatnonzero(np.linalg.norm(boxes[:,[0,2]]-target[[0,2]],axis=1) <= r+r0)
    iou = np.zeros(len(boxes))
    if len(indices):
        iou[indices] = d3_box_overlap(boxes[indices],target[None])[:,0]
    return iou


def summarize(iou, scores, boxes, ids=None):
    if not len(iou):
        return dict(count=0, above_iou=0, best_iou=0., score=None, camera_box=None)
    j = int(np.argmax(iou))
    out = dict(count=len(iou),above_iou=int((iou>THRESHOLD).sum()),
               best_iou=float(iou[j]),score=float(scores[j]),camera_box=boxes[j].tolist())
    if ids is not None:
        out['anchor_id'] = int(ids[j])
    return out


def trace(head, captured, p, anno, cases, metric, ped):
    inputs, nms, head_final = captured['head'], captured['nms'], captured['final']
    actual_boxes, bev, actual_scores, score_thr, max_num, cfg, actual_dirs = nms
    raw, scores, dirs, top = all_candidates(head, inputs, cfg)
    torch.testing.assert_close(raw[top], actual_boxes)
    torch.testing.assert_close(scores[top], actual_scores[:, :head.num_classes])
    torch.testing.assert_close(dirs[top], actual_dirs)
    raw[:,6] = (dense.limit_period(raw[:,6]-head.dir_offset,head.dir_limit_offset,np.pi)
                + head.dir_offset + np.pi*dirs.to(raw.dtype))
    cls = inputs['input_meta']['box_type_3d']
    transform = np.array(anno['images']['CAM2']['lidar2cam'], dtype=np.float32)
    camera = cls(raw,box_dim=head.box_code_size).convert_to(Box3DMode.CAM,transform).tensor.cpu().numpy()
    score = scores[:,ped].cpu().numpy()
    passed = actual_scores[:,ped] > score_thr
    score_ids = top[passed]
    nms_func = nms_bev if cfg.use_rotate_nms else nms_normal_bev
    kept_rel = nms_func(bev[passed],actual_scores[passed,ped],cfg.nms_thr) if passed.any() else score_ids
    nms_ids = score_ids[kept_rel]
    pred = p.pred_instances_3d
    formatted = metric.convert_valid_bboxes({k:v.cpu() for k,v in pred.items()},anno)
    def raw_ped(instance):
        mask = instance.labels_3d==ped
        boxes = instance.bboxes_3d[mask].convert_to(Box3DMode.CAM,transform).tensor.cpu().numpy()
        return boxes, instance.scores_3d[mask].cpu().numpy()
    mask = (formatted['label_preds']==ped).reshape(-1)
    image_boxes = formatted['bbox'][mask]
    cam = anno['images']['CAM2']
    heights = np.clip(image_boxes[:,3],0,cam['height']) - np.clip(image_boxes[:,1],0,cam['height'])
    final_boxes = formatted['box3d_camera'][mask]
    final_scores = formatted['scores'][mask]
    stages = {k:v.cpu().numpy() for k,v in [('topk',top),('score_filter',score_ids),('nms',nms_ids)]}
    details = []
    for case in cases:
        obj = anno['instances'][case['gt_index']]
        assert obj['bbox_label']==0 and abs(obj['bbox_3d'][2]-case['depth_m'])<1e-4
        target = np.asarray(obj['bbox_3d'],dtype=np.float32)
        iou = overlaps(camera,target)
        result = {'all_decoded':summarize(iou,score,camera,np.arange(len(raw)))}
        for key,ids in stages.items():
            result[key] = summarize(iou[ids],score[ids],camera[ids],ids)
        for key,instance in [('head_cap',head_final),('model_output',pred)]:
            boxes, ss = raw_ped(instance)
            result[key] = summarize(overlaps(boxes,target),ss,boxes)
        result['evaluator'] = summarize(overlaps(final_boxes,target),final_scores,final_boxes)
        valid = heights>=40
        result['evaluator_height'] = summarize(overlaps(final_boxes[valid],target),final_scores[valid],final_boxes[valid])
        loss = next((k for k,v in result.items() if not v['above_iou']),'retained')
        suppressor = None
        if loss=='nms':
            j = result['score_filter']['anchor_id']
            loc = (score_ids==j).nonzero().flatten().item()
            candidates = bev[passed]
            ss = actual_scores[passed,ped]
            for winner in kept_rel.tolist():
                if ss[winner] < ss[loc]:
                    break
                pair = torch.tensor([winner,loc],device=bev.device)
                if nms_func(candidates[pair],ss[pair],cfg.nms_thr).tolist()==[0]:
                    aid = int(score_ids[winner])
                    suppressor = dict(anchor_id=aid,score=float(score[aid]),iou_to_target=float(iou[aid]),camera_box=camera[aid].tolist())
                    break
        details.append(dict(**case,stages=result,first_without_qualified_box=loss,suppressor=suppressor))
    return details, formatted


def check_saved(formatted, saved, anno):
    # Compare each same-class score and camera box, independent of output order.
    names = np.array(['Pedestrian','Cyclist','Car'])
    labels = formatted['label_preds'].reshape(-1).astype(int)
    a = np.column_stack([labels,formatted['scores'],formatted['box3d_camera']])
    b_labels = [list(names).index(n) for n in saved['name']]
    b = np.column_stack([b_labels,saved['score'],saved['location'],saved['dimensions'],saved['rotation_y']])
    assert len(a)==len(b)
    if len(a):
        for key in (a,b):
            key[:,-1] = (key[:,-1]+np.pi)%(2*np.pi)-np.pi
        a = a[np.lexsort((-a[:,1],a[:,0]))]
        b = b[np.lexsort((-b[:,1],b[:,0]))]
        delta = a-b
        delta[:,-1] = (delta[:,-1]+np.pi)%(2*np.pi)-np.pi
        np.testing.assert_allclose(delta,0,atol=2e-3)
        return float(np.max(np.abs(delta)))
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-run',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    assert Path.cwd().resolve()==ROOT
    cases = json.loads((args.case_run/'head-trace-cases.json').read_text())
    assert len(cases)==24
    if args.smoke:
        lost = list(dict.fromkeys(x['frame'] for x in cases if x['group']=='lost'))[:2]
        kept = next(x['frame'] for x in cases if x['group']=='kept' and x['frame'] not in lost)
        cases = [x for x in cases if x['frame'] in [*lost,kept]]
    indices = sorted({x['frame'] for x in cases})
    with (WORK/'.gpu0-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_summary = json.loads((args.case_run/'summary.json').read_text())
        cp = MODELS_TO_RUN['gate_main']
        config = Path('configs/experiments/reliability_fusion_v2/gate_main.py')
        assert file_hash(cp)==run_summary['models']['gate_main']['sha256']
        assert file_hash(config)==run_summary['models']['gate_main']['config_sha256']
        assert file_hash(ANN)==run_summary['annotation_sha256']
        register_all_modules()
        cfg = Config.fromfile(config)
        cfg.test_dataloader.dataset.indices = indices
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
        loader = Runner.build_dataloader(cfg.test_dataloader,seed=0)
        classes = list(loader.dataset.metainfo['classes'])
        assert classes==['Pedestrian','Cyclist','Car']
        metric = METRICS.build(cfg.test_evaluator[0])
        annotations = load(ANN)['data_list']
        model = MODELS.build(cfg.model).cuda().eval()
        model.load_state_dict(torch.load(cp,map_location='cpu')['state_dict'],strict=True)
        head = model.bbox_head
        assert head.use_sigmoid_cls
        expected = {}
        for condition in ('clean','blur'):
            with (args.case_run/'gate_main'/condition/'pred_instances_3d.pkl').open('rb') as stream:
                expected[condition] = pickle.load(stream)
        args.work_dir.mkdir(parents=True,exist_ok=False)
        original = head._predict_by_feat_single
        nms_fn = dense.box3d_multiclass_nms
        captured = {}
        def capture_head(**kwargs):
            captured['head'] = kwargs
            result = original(**kwargs)
            captured['final'] = result
            return result
        def capture_nms(*argv):
            captured['nms'] = argv
            return nms_fn(*argv)
        records = []
        max_difference = 0.
        with patch.object(head,'_predict_by_feat_single',side_effect=capture_head), patch.object(dense,'box3d_multiclass_nms',side_effect=capture_nms), torch.inference_mode():
            for offset,batch in enumerate(loader):
                i = indices[offset]
                for condition in ('clean','blur'):
                    captured.clear()
                    predictions = model.test_step(corrupt_batch(copy.deepcopy(batch),condition))
                    align_sample_indices(predictions,indices,offset,annotations)
                    rows, formatted = trace(head,captured,predictions[0],annotations[i],
                                            [x for x in cases if x['frame']==i],metric,classes.index('Pedestrian'))
                    max_difference = max(max_difference,check_saved(formatted,expected[condition][i],annotations[i]))
                    records.extend(dict(condition=condition,**row) for row in rows)
                print('FRAME',i,offset+1,len(indices),flush=True)
        assert len(records)==2*len(cases) and file_hash(cp)==run_summary['models']['gate_main']['sha256']
        (args.work_dir/'summary.json').write_text(json.dumps(dict(frames=indices,smoke=args.smoke,max_final_difference=max_difference,records=records),indent=2,allow_nan=False),encoding='utf-8')
        print('COMPLETED',args.work_dir/'summary.json',flush=True)


if __name__=='__main__':
    main()
