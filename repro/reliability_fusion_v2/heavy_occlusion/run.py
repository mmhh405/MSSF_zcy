"""Compare three new deterministic occlusion masks under the same B settings."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine import load
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

sys.path.insert(0,str(Path.cwd()/'tools/analysis_tools'))
import reliability_robustness as robust

MASKS = {
    'left_25': (.05,.25,.55,.75),
    'center_12': (.325,.325,.675,.675),
    'center_42': (.175,.175,.825,.825),
}
MODEL_CONFIGS = {'control':'control_aug','v2':'gate_main'}


def mask_image(image, condition):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'Expected CHW image, got {image.shape}')
    h,w = image.shape[-2:]
    x0,y0,x1,y1 = MASKS[condition]
    out = image.clone()
    out[:,round(y0*h):round(y1*h),round(x0*w):round(x1*w)] = 0
    return out


def main():
    root = Path(sys.argv[1])
    samples = int(sys.argv[2])
    out = root/('smoke' if samples<1296 else 'full')
    out.mkdir(exist_ok=False)
    ann = Path('data/vod/kitti_infos_val.pkl')
    annotations = load(ann)['data_list']
    assert len(annotations)==1296
    indices = robust.uniform_indices(len(annotations),samples)
    training = json.loads(Path(__file__).with_name('checkpoints.json').read_text())
    register_all_modules()
    MMLogger.get_instance('new_occlusions',log_file=str(out/'metrics.log'))
    robust.corrupt_image = mask_image
    summary = dict(valid=False,samples=samples,indices=indices,mask_xyxy_fraction=MASKS,
                   annotation_sha256=robust.file_hash(ann),nms_pre=300,score_thr=.1,seeds={})
    for seed in ((0,) if samples<1296 else (0,1,2)):
        summary['seeds'][seed] = {}
        for label,variant in MODEL_CONFIGS.items():
            target = out/f'seed{seed}'/label
            target.mkdir(parents=True)
            entry = next(x for x in training if x['training_seed']==seed and x['variant']==variant)
            checkpoint = Path(entry['path'])
            assert robust.file_hash(checkpoint)==entry['sha256']
            cfg=Config.fromfile(f'configs/experiments/reliability_fusion_v2/{variant}.py')
            cfg.model.test_cfg.nms_pre=300
            cfg.model.test_cfg.score_thr=.1
            cfg.test_dataloader.dataset.indices=indices
            cfg.test_dataloader.num_workers=0
            cfg.test_dataloader.persistent_workers=False
            cfg.dump(str(target/'resolved.py'))
            torch.manual_seed(0)
            model=MODELS.build(cfg.model).cuda().eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
            loader=Runner.build_dataloader(cfg.test_dataloader,seed=0)
            rows={}
            summary['seeds'][seed][label]=dict(checkpoint_sha256=entry['sha256'],conditions=rows)
            for c in MASKS:
                print('START',seed,label,c,flush=True)
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
                        data=robust.corrupt_batch(data,c)
                        pred=model.test_step(data)
                        offset=robust.align_sample_indices(pred,indices,offset,annotations)
                        evaluator.process(data_samples=pred,data_batch=data)
                        if offset%128==0:
                            print('PROGRESS',seed,label,c,offset,samples,flush=True)
                assert offset==samples
                result=evaluator.evaluate(offset)
                suffixes={'eaa':'entire_area/map','dc':'roi/map','pedestrian_ap':'entire_area/Pedestrian_3d_all'}
                row={key:float(next(v for k,v in result.items() if k.endswith(suffix))) for key,suffix in suffixes.items()}
                row['metrics']={k:float(v) for k,v in result.items()}
                rows[c]=row
                (out/'progress.json').write_text(json.dumps(summary,indent=2))
                print('RESULT',seed,label,c,{k:row[k] for k in suffixes},flush=True)
            assert robust.file_hash(checkpoint)==entry['sha256']
            del model
            torch.cuda.empty_cache()
    summary['valid']=True
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETED',out/'summary.json',flush=True)


if __name__=='__main__':
    main()
