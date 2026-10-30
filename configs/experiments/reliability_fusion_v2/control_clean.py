"""V2 clean control: original architecture, same optimizer and normalization policy."""
_base_ = '../reliability_fusion/mssf_pp_vod_control.py'
work_dir = 'work_dirs/reliability_fusion_v2/control_clean'
model = dict(freeze_norm_stats=True, gate_clean_weight=0.)
train_pipeline = _base_.train_dataloader.dataset.dataset.pipeline
train_pipeline.insert(2, dict(type='RandomImageDegradation', prob=0.))
train_pipeline[-1]['type'] = 'PackReliabilityInputs'
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'reliability_gate': dict(lr_mult=10.),
    'query_reliability_gate': dict(lr_mult=10.)}))
custom_hooks = [dict(type='ReliabilityExperimentHook',
                     readme='README_reliability_fusion_v2.md',
                     results='repro/reliability_fusion_v2/results.json')]
reliability_experiment = dict(variant='v2-control-clean', purpose='finetune')
