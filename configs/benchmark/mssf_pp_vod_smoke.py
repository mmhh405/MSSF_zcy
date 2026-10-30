# First 16 validation frames, only for checking the inference/evaluation pipeline.
_base_ = './mssf_pp_vod.py'
work_dir = 'work_dirs/baseline_mssf_pp_vod_smoke'

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=list(range(16))),
)
val_dataloader = test_dataloader
test_evaluator = [dict(
    type='VODMetric',
    ann_file='data/vod/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    pklfile_prefix=work_dir + '/predictions',
    submission_prefix=None,
)]
val_evaluator = test_evaluator
