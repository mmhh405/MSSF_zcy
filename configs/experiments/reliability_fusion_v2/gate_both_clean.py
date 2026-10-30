"""Both query and attention gates, clean-only anchor, no image degradation."""
_base_ = './gate_attention_clean.py'
work_dir = 'work_dirs/reliability_fusion_v2/gate_both_clean'
model = dict(
    gate_clean_weight=.1,
    middle_encoder=dict(optional_cfg=dict(query_reliability_gate=dict(
        type='ReliabilityGateV2', enabled=True, hidden_channels=16))),
    backbone=dict(optional_cfg=dict(query_reliability_gate=dict(
        type='ReliabilityGateV2', enabled=True, hidden_channels=16))))
reliability_experiment = dict(variant='v2-both-clean-anchor', purpose='finetune')
