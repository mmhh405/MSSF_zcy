"""Two-stage reliability gates; same training budget as the control config."""
_base_ = './mssf_pp_vod_control.py'
work_dir = 'work_dirs/reliability_fusion/gate'
model = dict(
    middle_encoder=dict(optional_cfg=dict(reliability_gate=dict(
        enabled=True, hidden_channels=16, init_probability=0.99))),
    backbone=dict(optional_cfg=dict(reliability_gate=dict(
        enabled=True, hidden_channels=16, init_probability=0.99))))
reliability_experiment = dict(variant='reliability-gate', purpose='finetune')
