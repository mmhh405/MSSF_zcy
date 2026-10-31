#!/usr/bin/env bash
# Run selected seed-0 ablations sequentially; defaults preserve the original pair.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
BATCH="${1:-$ROOT/work_dirs/reliability_fusion_v2/ablation/$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then
  VARIANTS=("$@")
else
  VARIANTS=(gate_aug_attention_only gate_aug_no_anchor)
fi
for VARIANT in "${VARIANTS[@]}"; do
  case "$VARIANT" in
    gate_aug_attention_only|gate_aug_no_anchor|gate_aug_query_only) ;;
    *) echo "Unsupported ablation: $VARIANT" >&2; exit 2 ;;
  esac
done
case "$BATCH" in "$ROOT"/work_dirs/reliability_fusion_v2/ablation/*) ;; *) echo 'Batch path must be inside this project ablation directory' >&2; exit 2 ;; esac
exec 9>"$ROOT/work_dirs/reliability_fusion_v2/.gpu0-queue.lock"
if ! flock -n 9; then
  echo 'Another reliability queue holds GPU 0 lock. This duplicate launch has stopped.' >&2
  exit 75
fi
if test -n "$(git status --porcelain)"; then
  echo 'Record current working-tree changes before starting the ablation queue.' >&2
  exit 2
fi
mkdir -p "$(dirname "$BATCH")"
mkdir "$BATCH"
CODE="$(git rev-parse HEAD)"
printf '%s\n' "$CODE" > "$BATCH/code_commit.txt"
git ls-files -z '*.py' '*.sh' | xargs -0 sha256sum > "$BATCH/source.sha256"
export CUDA_VISIBLE_DEVICES=0 NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1
CONTROL="$ROOT/work_dirs/reliability_fusion_v2/control_aug/20261030_214524/epoch_6.pth"
test -s "$CONTROL"
TASK='initialization'
on_error() { local code=$?; echo "$(date --iso-8601=seconds) FAILED task=$TASK exit=$code; inspect the task console.log" >&2; exit "$code"; }
trap on_error ERR
for VARIANT in "${VARIANTS[@]}"; do
  test "$(git rev-parse HEAD)" = "$CODE"
  sha256sum --status -c "$BATCH/source.sha256"
  RUN="$BATCH/$VARIANT"
  mkdir "$RUN"
  TASK="train:$VARIANT"
  echo "$(date --iso-8601=seconds) START $TASK work_dir=$RUN"
  python -u "$ROOT/tools/train.py" \
    "$ROOT/configs/experiments/reliability_fusion_v2/${VARIANT}.py" \
    --work-dir "$RUN" --cfg-options randomness.seed=0 \
    reliability_experiment.variant="v2-${VARIANT}-seed0" \
    > "$RUN/console.log" 2>&1
  python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'reliability_result.json').read_text())['summary']
assert s['training_completed'] and s['random_seed']==0 and s['validation_epoch']==6
assert s['iterations']==30834 and s['samples']==1296 and (p/'epoch_6.pth').is_file()
PY
  echo "$(date --iso-8601=seconds) TRAIN_DONE $VARIANT"
  test "$(git rev-parse HEAD)" = "$CODE"
  sha256sum --status -c "$BATCH/source.sha256"
  EVAL="$RUN/robustness"
  mkdir "$EVAL"
  TASK="evaluate:$VARIANT"
  echo "$(date --iso-8601=seconds) START $TASK"
  python -u "$ROOT/tools/analysis_tools/reliability_robustness.py" \
    --samples 1296 --conditions clean dark blur occlusion \
    --control-config "$ROOT/configs/experiments/reliability_fusion_v2/control_aug.py" \
    --gate-config "$ROOT/configs/experiments/reliability_fusion_v2/${VARIANT}.py" \
    --control-checkpoint "$CONTROL" --gate-checkpoint "$RUN/epoch_6.pth" \
    --readme "$ROOT/README_reliability_fusion_v2.md" \
    --archive-dir "$ROOT/repro/reliability_fusion_v2/robustness" \
    --work-dir "$EVAL" > "$EVAL/console.log" 2>&1
  python - "$EVAL/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));assert s['valid'] and s['full_validation']
assert s['samples']==1296 and len(s['results'])==8
PY
  echo "$(date --iso-8601=seconds) EVAL_DONE $VARIANT"
done
echo "$(date --iso-8601=seconds) ALL_COMPLETED $BATCH"
