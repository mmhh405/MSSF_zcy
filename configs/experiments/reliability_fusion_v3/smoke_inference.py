"""Four validation frames; wiring check, not an accuracy experiment."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v3/smoke_inference'
test_dataloader = dict(num_workers=0, persistent_workers=False,
                       dataset=dict(indices=list(range(4))))
val_dataloader = test_dataloader
reliability_experiment = dict(variant='v3-four-frame-forward', purpose='smoke')
