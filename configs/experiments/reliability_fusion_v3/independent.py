"""Matched budgets with independent, single-modality selection scores."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/independent'
model = dict(early_budget=dict(router=dict(use_radar_to_image=False,
                                         use_image_to_radar=False)))
reliability_experiment = dict(variant='v3-independent-budget', purpose='finetune')
