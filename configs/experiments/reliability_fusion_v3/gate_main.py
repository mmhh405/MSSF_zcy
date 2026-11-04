"""V3 candidate: early bidirectional compute selection + existing V2 fusion."""
_base_ = '../reliability_fusion_v2/gate_main.py'
custom_imports = dict(imports=['mmdet3d.models.detectors.voxelnet_fuse_v3'],
                      allow_failed_imports=False)
work_dir = 'work_dirs/reliability_fusion_v3/gate_main'
# This is a warm start, not a trained V3 checkpoint. Override for another seed.
load_from = 'work_dirs/reliability_fusion_v2/gate_main/20261030_214524/epoch_6.pth'
model = dict(type='VoxelNetFuseV3', early_budget=dict(
    mode='select',
    router=dict(preview_size=(160, 256), hidden_channels=16, tile_size=256,
                image_keep_ratio=.2, radar_keep_ratio=.8, reserve_fraction=.25,
                min_pillars=32, use_radar_to_image=True, use_image_to_radar=True,
                supervision_weight=.2, budget_weight=.01),
    image_executor=dict(global_scale=.5, halo=32, tile_batch_size=2, blend=.5)),
    test_cfg=dict(nms_pre=300, score_thr=.1))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'reliability_gate': dict(lr_mult=10.),
    'query_reliability_gate': dict(lr_mult=10.),
    'early_router': dict(lr_mult=10.)}))
custom_hooks = [dict(type='ReliabilityExperimentHook',
                     readme='README_reliability_fusion_v3.md',
                     results='repro/reliability_fusion_v3/results.json')]
reliability_experiment = dict(variant='v3-bidirectional-budget', purpose='finetune')
