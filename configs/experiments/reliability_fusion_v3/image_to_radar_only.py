"""Matched budgets; disable only the radar-to-image cross-modal prior."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/image_to_radar_only'
model = dict(early_budget=dict(router=dict(use_radar_to_image=False)))
reliability_experiment = dict(variant='v3-image-to-radar-only', purpose='finetune')
