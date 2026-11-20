"""Matched six-epoch screening protocol for the spatial image router."""
_base_ = '../reliability_fusion_v3/gate_image_budget_only.py'
load_from = '/home/d514/桌面/zcy/MSSF_zcy/work_dirs/reliability_fusion_v2/gate_main/20261030_214524/epoch_6.pth'
custom_hooks = []
work_dir = 'work_dirs/reliability_fusion_v31/legacy_seed0'
reliability_experiment = dict(variant='v31-legacy', purpose='screening')
