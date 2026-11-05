"""Candidate for precision recovery; train afresh from the same V2 warm start.

The first V3 used radar=.8/global=.5. Fixed-weight diagnostics motivated this
more conservative budget, but do not establish the result after fine-tuning.
"""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/gate_quality_first'
model = dict(early_budget=dict(
    router=dict(radar_keep_ratio=.95),
    image_executor=dict(global_scale=.625)))
reliability_experiment = dict(variant='v3-quality-first-radar95-global0625',
                              purpose='finetune')
