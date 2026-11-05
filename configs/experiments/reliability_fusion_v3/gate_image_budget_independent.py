"""Budget-matched early-routing ablation; V2 fusion gates remain enabled."""
_base_ = './gate_image_budget_only.py'
work_dir = 'work_dirs/reliability_fusion_v3/gate_image_budget_independent'
model = dict(early_budget=dict(router=dict(
    use_radar_to_image=False, use_image_to_radar=False)))
reliability_experiment = dict(
    variant='v3-image-budget-independent-radar100-global0625', purpose='finetune')
