"""Matched six-epoch candidate with quarter-strength local feature mixing."""
_base_ = '../reliability_fusion_v31/legacy.py'
work_dir = 'work_dirs/reliability_fusion_v31/legacy_blend025_seed0'
model = dict(early_budget=dict(image_executor=dict(blend=.25)))
