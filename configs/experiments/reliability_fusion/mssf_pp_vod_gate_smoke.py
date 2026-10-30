"""Sixteen-frame inference check using untrained gates, not paper accuracy."""
_base_ = './mssf_pp_vod_gate.py'
work_dir = 'work_dirs/reliability_fusion/gate_smoke'
test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False,
                       dataset=dict(indices=list(range(16))))
val_dataloader = test_dataloader
reliability_experiment = dict(variant='gate-untrained-inference', purpose='smoke')
