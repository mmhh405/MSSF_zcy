"""Matched fine-tuning control; original architecture, no reliability gate."""
_base_ = '../../benchmark/mssf_pp_vod.py'

work_dir = 'work_dirs/reliability_fusion/control'
train_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True,
                        dataset=dict(times=1))
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)
optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [dict(type='CosineAnnealingLR', T_max=6, eta_min=1e-6, by_epoch=True)]
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))
randomness = dict(seed=0, deterministic=False)
auto_scale_lr = dict(enable=False)
test_evaluator = [dict(type='VODMetric', ann_file='data/vod/kitti_infos_val.pkl',
                       backend_args=None, metric='bbox', pklfile_prefix=None,
                       submission_prefix=None)]
val_evaluator = test_evaluator
custom_hooks = [dict(type='ReliabilityExperimentHook')]
reliability_experiment = dict(variant='control-finetune', purpose='finetune')
