"""Matched budgets; disable only the image-to-radar cross-modal prior."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/radar_to_image_only'
model = dict(early_budget=dict(router=dict(use_image_to_radar=False)))
reliability_experiment = dict(variant='v3-radar-to-image-only', purpose='finetune')
