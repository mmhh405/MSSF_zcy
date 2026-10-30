# Official MSSF-PP/VoD checkpoint, inference-only baseline configuration.
_base_ = '../../exps/mssf_pp_vod/pointpillars_hv_secfpn_8xb6-160e_vod-3d-3class_mm_bb_bevfeat_lr0001.py'

load_from = 'exps/mssf_pp_vod/epoch_40.pth'
work_dir = 'work_dirs/baseline_mssf_pp_vod'

# The full checkpoint supplies both image modules; no separate initialization file is required.
model = dict(img_backbone=dict(init_cfg=None), img_neck=dict(init_cfg=None))
test_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
val_dataloader = test_dataloader

# Keep a valid hook config; drawing is disabled for accuracy evaluation.
default_hooks = dict(visualization=dict(
    type='Det3DVisualizationHook', draw=False, output_detsample_dir=None))
test_evaluator = [dict(
    type='VODMetric',
    ann_file='data/vod/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    pklfile_prefix=work_dir + '/predictions',
    submission_prefix=None,
)]
val_evaluator = test_evaluator
