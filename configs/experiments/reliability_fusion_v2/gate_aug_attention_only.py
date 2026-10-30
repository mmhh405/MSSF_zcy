"""Same image degradation/anchor as main; query image inputs remain ungated."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v2/gate_aug_attention_only'
model = dict(
    middle_encoder=dict(optional_cfg=dict(query_reliability_gate=dict(enabled=False))),
    backbone=dict(optional_cfg=dict(query_reliability_gate=dict(enabled=False))))
reliability_experiment = dict(variant='v2-aug-attention-only', purpose='finetune')
