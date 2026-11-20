"""Same budget and training as legacy, with explicit image coverage policy."""
_base_ = './legacy.py'
work_dir = 'work_dirs/reliability_fusion_v31/spatial_seed0'
model = dict(early_budget=dict(router=dict(image_selection='spatial')))
reliability_experiment = dict(variant='v31-spatial', purpose='screening')
