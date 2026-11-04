"""Same warm start/augmentation/epochs; no early selection, V2 fusion retained."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/dense_control'
model = dict(early_budget=dict(mode='dense'))
reliability_experiment = dict(variant='v3-dense-v2-control', purpose='finetune')
