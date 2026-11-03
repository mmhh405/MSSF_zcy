_base_ = './vx_mm_r50_ablation_n1m5.py'

model = dict(
    pts_middle_encoder=dict(
        optional_cfg=dict(geometry_prior=True, semantic_prior=True)))
