"""Parameterization ablation: V2 attention gates only, no image degradation/anchor."""
_base_ = './control_clean.py'
work_dir = 'work_dirs/reliability_fusion_v2/gate_attention_clean'
model = dict(
    middle_encoder=dict(optional_cfg=dict(reliability_gate=dict(
        type='ReliabilityGateV2', enabled=True, hidden_channels=16))),
    backbone=dict(optional_cfg=dict(reliability_gate=dict(
        type='ReliabilityGateV2', enabled=True, hidden_channels=16))))
reliability_experiment = dict(variant='v2-attention-clean', purpose='finetune')
