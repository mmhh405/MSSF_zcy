"""Augmentation-only control: isolates gains from synthetic degradation exposure."""
_base_ = './control_clean.py'
work_dir = 'work_dirs/reliability_fusion_v2/control_aug'
train_pipeline = [dict(step, prob=.5) if step['type'] == 'RandomImageDegradation'
                  else step for step in _base_.train_dataloader.dataset.dataset.pipeline]
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
reliability_experiment = dict(variant='v2-control-aug', purpose='finetune')
