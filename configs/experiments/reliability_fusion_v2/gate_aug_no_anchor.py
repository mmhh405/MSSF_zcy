"""Same as V2 main, with the clean gate regularizer disabled."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v2/gate_aug_no_anchor'
model = dict(gate_clean_weight=0.)
reliability_experiment = dict(variant='v2-aug-no-anchor', purpose='finetune')
