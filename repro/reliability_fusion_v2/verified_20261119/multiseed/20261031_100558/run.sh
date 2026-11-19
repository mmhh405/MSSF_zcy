#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export PYTHONDONTWRITEBYTECODE=1

exec 9>"$ROOT/work_dirs/reliability_fusion_v2/.gpu0-queue.lock"
flock -n 9

CODE="$(git rev-parse HEAD)"
printf '%s\n' "$CODE" > "$BATCH/code_commit.txt"
git ls-files -z '*.py' | xargs -0 sha256sum > "$BATCH/source.sha256"

for SEED in 1 2; do
  for MODEL in control_aug gate_main; do
    test "$(git rev-parse HEAD)" = "$CODE"
    sha256sum --status -c "$BATCH/source.sha256"

    RUN="$BATCH/seed${SEED}/${MODEL}"
    mkdir -p "$RUN"
    test ! -e "$RUN/epoch_6.pth"

    echo "START seed=$SEED model=$MODEL"
    python -u "$ROOT/tools/train.py" \
      "$ROOT/configs/experiments/reliability_fusion_v2/${MODEL}.py" \
      --work-dir "$RUN" \
      --cfg-options \
        randomness.seed="$SEED" \
        reliability_experiment.variant="v2-${MODEL}-seed${SEED}" \
      > "$RUN/console.log" 2>&1

    test -s "$RUN/epoch_6.pth"
    echo "TRAIN_DONE seed=$SEED model=$MODEL"
  done

  EVAL="$BATCH/seed${SEED}/robustness"
  mkdir -p "$EVAL"

  python -u "$ROOT/tools/analysis_tools/reliability_robustness.py" \
    --samples 1296 \
    --conditions clean dark blur occlusion \
    --control-config "$ROOT/configs/experiments/reliability_fusion_v2/control_aug.py" \
    --gate-config "$ROOT/configs/experiments/reliability_fusion_v2/gate_main.py" \
    --control-checkpoint "$BATCH/seed${SEED}/control_aug/epoch_6.pth" \
    --gate-checkpoint "$BATCH/seed${SEED}/gate_main/epoch_6.pth" \
    --readme "$ROOT/README_reliability_fusion_v2.md" \
    --archive-dir "$ROOT/repro/reliability_fusion_v2/robustness" \
    --work-dir "$EVAL" \
    > "$EVAL/console.log" 2>&1

  echo "EVAL_DONE seed=$SEED"
done

echo "ALL_COMPLETED"
