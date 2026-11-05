"""Ablation candidate: retain all radar geometry, route high-resolution images.

Radar-to-image priors and V2 fusion remain. This is not evidence that physical
radar pruning is beneficial; use it to isolate the cost of deleting radar points.
"""
_base_ = './gate_quality_first.py'
work_dir = 'work_dirs/reliability_fusion_v3/gate_image_budget_only'
model = dict(early_budget=dict(router=dict(radar_keep_ratio=1.)))
reliability_experiment = dict(variant='v3-image-budget-only-radar100-global0625',
                              purpose='finetune')
