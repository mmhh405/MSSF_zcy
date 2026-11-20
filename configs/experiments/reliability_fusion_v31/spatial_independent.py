"""Same spatial policy and image budget, with both early cross priors disabled."""
_base_ = './spatial.py'
work_dir = 'work_dirs/reliability_fusion_v31/spatial_independent_seed0'
model = dict(early_budget=dict(router=dict(
    use_radar_to_image=False, use_image_to_radar=False)))
reliability_experiment = dict(variant='v31-spatial-independent', purpose='screening')
