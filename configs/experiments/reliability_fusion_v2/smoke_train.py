"""Four optimizer updates; integration check, not a trained checkpoint."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v2/smoke_train'
train_dataloader = dict(num_workers=0, persistent_workers=False,
                        dataset=dict(dataset=dict(indices=list(range(4)))))
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=4, val_interval=10000)
param_scheduler = None
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(logger=dict(interval=1),
                     checkpoint=dict(by_epoch=False, interval=4, max_keep_ckpts=1))
reliability_experiment = dict(variant='v2-four-iteration-backward', purpose='smoke')
