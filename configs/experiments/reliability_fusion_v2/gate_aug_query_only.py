"""Keep query-image gates; remove attention-output gates at both stages."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v2/gate_aug_query_only'
model = dict(
    middle_encoder=dict(optional_cfg=dict(reliability_gate=dict(enabled=False))),
    backbone=dict(optional_cfg=dict(reliability_gate=dict(enabled=False))))
reliability_experiment = dict(variant='v2-aug-query-only', purpose='finetune')
