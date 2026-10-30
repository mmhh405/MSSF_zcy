"""Two optimizer updates on four training frames to check backward integration."""
_base_ = './mssf_pp_vod_gate.py'
work_dir = 'work_dirs/reliability_fusion/train_smoke'
train_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False,
                        dataset=dict(dataset=dict(indices=list(range(4)))))
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=2,
                 val_interval=10000)
param_scheduler = None
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(logger=dict(interval=1),
                     checkpoint=dict(by_epoch=False, interval=2, max_keep_ckpts=1))
reliability_experiment = dict(variant='gate-two-iteration-backward', purpose='smoke')
