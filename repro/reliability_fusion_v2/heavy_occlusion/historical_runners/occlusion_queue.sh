#!/usr/bin/env bash
set -euo pipefail
cd '/home/d514/桌面/zcy/MSSF_zcy'
batch="$1"
runner="$2"
export CUDA_VISIBLE_DEVICES=0 NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONPATH="$PWD"
while nvidia-smi -i 0 --query-compute-apps=process_name --format=csv,noheader | grep -qi 'python'; do
    printf 'WAIT_GPU %s\n' "$(date --iso-8601=seconds)"
    sleep 30
done
exec flock -w 600 work_dirs/reliability_fusion_v2/.gpu0-queue.lock bash -c '
set -e
"$1" -u "$2/run.py" "$2" 32
"$1" -u "$2/run.py" "$2" 1296
echo "ALL_COMPLETED $2/full/summary.json"
' _ "$runner" "$batch"
