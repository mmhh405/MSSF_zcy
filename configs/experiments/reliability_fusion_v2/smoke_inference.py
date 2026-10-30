"""First 16 validation frames, untrained identity-initialized V2 gates."""
_base_ = './gate_main.py'
work_dir = 'work_dirs/reliability_fusion_v2/smoke_inference'
test_dataloader = dict(num_workers=0, persistent_workers=False,
                       dataset=dict(indices=list(range(16))))
val_dataloader = test_dataloader
reliability_experiment = dict(variant='v2-untrained-inference', purpose='smoke')
