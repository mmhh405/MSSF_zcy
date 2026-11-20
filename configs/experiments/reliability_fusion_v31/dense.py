"""Full image computation with the matched V2 initialization and training."""
_base_ = './legacy.py'
work_dir = 'work_dirs/reliability_fusion_v31/dense_seed0'
model = dict(early_budget=dict(mode='dense'))
reliability_experiment = dict(variant='v31-dense', purpose='screening')
