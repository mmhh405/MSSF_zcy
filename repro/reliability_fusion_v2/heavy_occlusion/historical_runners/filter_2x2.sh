#!/usr/bin/env bash
set -euo pipefail

cd '/home/d514/桌面/zcy/MSSF_zcy'
batch="$(cd -- "$(dirname -- "$0")" && pwd)"
samples="$1"
phase="$2"
weights='work_dirs/reliability_fusion_v2/gate_main/20261030_214524/epoch_6.pth'

export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export PYTHONDONTWRITEBYTECODE=1

run_pair() {
  python -u tools/analysis_tools/reliability_robustness.py \
    --samples "$samples" --conditions clean blur occlusion \
    --control-checkpoint "$weights" --gate-checkpoint "$weights" \
    --control-config "$batch/$1.py" --gate-config "$batch/$2.py" \
    --work-dir "$batch/$phase/$1$2" --no-record-readme
}

run_pair A B
run_pair C D
echo "ALL_COMPLETED $batch/$phase"
