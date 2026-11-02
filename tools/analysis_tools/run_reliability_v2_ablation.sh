#!/usr/bin/env bash
# Run selected component ablations; default is the original seed-0 pair.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
BATCH="${1:-$ROOT/work_dirs/reliability_fusion_v2/ablation/$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then VARIANTS=("$@"); else VARIANTS=(gate_aug_attention_only gate_aug_no_anchor); fi
for VARIANT in "${VARIANTS[@]}"; do
  case "$VARIANT" in
    gate_aug_attention_only|gate_aug_no_anchor|gate_aug_query_only) ;;
    *) echo "Unsupported ablation: $VARIANT" >&2; exit 2 ;;
  esac
done
read -r -a SEEDS <<< "${RELIABILITY_SEEDS:-0}"
if (( ${#SEEDS[@]} == 0 )); then echo 'Specify at least one training seed' >&2; exit 2; fi
SEEN=' '
for SEED in "${SEEDS[@]}"; do
  case "$SEED" in 0|1|2) ;; *) echo "Unsupported seed: $SEED" >&2; exit 2 ;; esac
  if [[ "$SEEN" == *" $SEED "* ]]; then echo "Duplicate seed: $SEED" >&2; exit 2; fi
  SEEN+="$SEED "
done
control_path() {
  case "$1" in
    0) printf '%s\n' "$ROOT/work_dirs/reliability_fusion_v2/control_aug/20261030_214524/epoch_6.pth" ;;
    1|2) printf '%s\n' "$ROOT/work_dirs/reliability_fusion_v2/multiseed/20261031_100513/seed$1/control_aug/epoch_6.pth" ;;
  esac
}
case "$BATCH" in "$ROOT"/work_dirs/reliability_fusion_v2/ablation/*) ;; *) echo 'Batch path must be inside this project ablation directory' >&2; exit 2 ;; esac
for SEED in "${SEEDS[@]}"; do test -s "$(control_path "$SEED")"; done
if [[ "${RELIABILITY_DRY_RUN:-0}" == 1 ]]; then
  for SEED in "${SEEDS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
      echo "PLAN seed=$SEED variant=$VARIANT control=$(control_path "$SEED") batch=$BATCH"
    done
  done
  exit 0
fi
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
printf '%s\n' "${SEEDS[@]}" > "$BATCH/training_seeds.txt"
printf '%s\n' "${VARIANTS[@]}" > "$BATCH/variants.txt"
git ls-files -z '*.py' '*.sh' | xargs -0 sha256sum > "$BATCH/source.sha256"
export CUDA_VISIBLE_DEVICES=0 NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1
TASK='initialization'
on_error() { local code=$?; echo "$(date --iso-8601=seconds) FAILED task=$TASK exit=$code; inspect the task console.log" >&2; exit "$code"; }
trap on_error ERR
for SEED in "${SEEDS[@]}"; do
  CONTROL="$(control_path "$SEED")"
  if (( ${#SEEDS[@]} == 1 )) && [[ "$SEED" == 0 ]]; then
    SEED_DIR="$BATCH"
  else
    SEED_DIR="$BATCH/seed${SEED}"
    mkdir "$SEED_DIR"
  fi
  for VARIANT in "${VARIANTS[@]}"; do
    test "$(git rev-parse HEAD)" = "$CODE"
    sha256sum --status -c "$BATCH/source.sha256"
    RUN="$SEED_DIR/$VARIANT"
    mkdir "$RUN"
    TASK="train:seed${SEED}:$VARIANT"
    echo "$(date --iso-8601=seconds) START $TASK work_dir=$RUN"
    python -u "$ROOT/tools/train.py" \
      "$ROOT/configs/experiments/reliability_fusion_v2/${VARIANT}.py" \
      --work-dir "$RUN" --cfg-options randomness.seed="$SEED" \
      reliability_experiment.variant="v2-${VARIANT}-seed${SEED}" \
      > "$RUN/console.log" 2>&1
    python - "$RUN" "$SEED" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'reliability_result.json').read_text())['summary']
assert s['training_completed'] and s['random_seed']==int(sys.argv[2]) and s['validation_epoch']==6
assert s['iterations']==30834 and s['samples']==1296 and (p/'epoch_6.pth').is_file()
PY
    echo "$(date --iso-8601=seconds) TRAIN_DONE seed=$SEED $VARIANT"
    test "$(git rev-parse HEAD)" = "$CODE"
    sha256sum --status -c "$BATCH/source.sha256"
    EVAL="$RUN/robustness"
    mkdir "$EVAL"
    TASK="evaluate:seed${SEED}:$VARIANT"
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
    echo "$(date --iso-8601=seconds) EVAL_DONE seed=$SEED $VARIANT"
  done
done
echo "$(date --iso-8601=seconds) ALL_COMPLETED $BATCH"
