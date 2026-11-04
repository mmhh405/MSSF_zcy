#!/usr/bin/env bash
# Run the two matched V3 experiments using an explicit Python interpreter.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
SEED=0
BASE=""
while (( $# )); do
    case "$1" in
        --work-dir) BASE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Usage: bash $0 [--work-dir DIR] [--seed INTEGER]" >&2; exit 2 ;;
    esac
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'Seed must be a nonnegative integer' >&2; exit 2; }
if [[ -n "${MSSF_PYTHON:-}" ]]; then
    PYTHON="$MSSF_PYTHON"
elif [[ -x "$ROOT/.runtime/mssf-bench/bin/python" ]]; then
    PYTHON="$ROOT/.runtime/mssf-bench/bin/python"
else
    PYTHON="/home/d514/miniconda3/envs/mssf-bench/bin/python"
fi
[[ -x "$PYTHON" ]] || { echo "Missing environment Python: $PYTHON" >&2; exit 1; }
export PATH="$(dirname -- "$PYTHON"):/home/d514/.local/bin:$PATH"
export PYTHONPATH="$ROOT"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/work_dirs/reliability_fusion_v3"
exec 9> "$ROOT/work_dirs/reliability_fusion_v3/.paired-training.lock"
flock -n 9 || { echo 'Another V3 paired queue is already running' >&2; exit 1; }
if [[ -z "$BASE" ]]; then
    BASE="$ROOT/work_dirs/reliability_fusion_v3/paired_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
fi
BASE="$(realpath -m -- "$BASE")"
[[ "$BASE" == "$ROOT/work_dirs/reliability_fusion_v3/"* ]] || {
    echo 'Work directory must be inside this project work_dirs/reliability_fusion_v3' >&2; exit 1;
}
mkdir -p "$BASE"
[[ ! -e "$BASE/run_manifest.json" ]] || { echo "Run directory already used: $BASE" >&2; exit 1; }
CURRENT_MODEL=preflight
trap 'code=$?; if (( code != 0 )); then echo "FAILED $CURRENT_MODEL exit=$code"; fi' EXIT

"$PYTHON" - "$ROOT" "$BASE" "$SEED" <<'PY'
import hashlib, json, pathlib, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo
import mmengine
import torch
from mmengine.config import Config

root, base, seed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
assert torch.cuda.is_available(), 'CUDA is not available in the selected environment'

def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()

configs, checkpoints = {}, {}
for model in ('dense_control', 'gate_main'):
    name = 'configs/experiments/reliability_fusion_v3/' + model + '.py'
    cfg = Config.fromfile(str(root / name))
    assert cfg.train_cfg.max_epochs == 6, 'Expected the matched six-epoch protocol'
    assert cfg.train_dataloader.batch_size == 1, 'Unexpected batch-size change'
    checkpoint = pathlib.Path(cfg.load_from)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    assert checkpoint.is_file(), checkpoint
    configs[model] = name
    checkpoints[str(checkpoint)] = digest(checkpoint)
assert len(checkpoints) == 1, 'Both models must use the same warm-start checkpoint'
for name in ('kitti_infos_train.pkl', 'kitti_infos_val.pkl'):
    assert (root / 'data/vod' / name).is_file(), name
tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '*.py'])
paths = [pathlib.Path(name.decode()) for name in tracked.split(b'\0') if name]
paths.append(pathlib.Path('tools/analysis_tools/run_reliability_v3_pair.sh'))
record = dict(started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    root=str(root), python=sys.executable, mmengine=mmengine.__version__,
    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
    seed=seed, epochs=6, batch_size=1, configs=configs, warm_start_sha256=checkpoints,
    git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    git_status=subprocess.check_output(['git', '-C', str(root), 'status', '--short'], text=True),
    source_sha256={str(path): digest(root / path) for path in paths})
(base / 'run_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('PREFLIGHT_OK python=' + sys.executable, flush=True)
print('WORK_DIR', base, flush=True)
PY

printf '%s\n' "$BASE" > "$ROOT/work_dirs/reliability_fusion_v3/last_paired_path.txt"
for CURRENT_MODEL in dense_control gate_main; do
    "$PYTHON" - "$ROOT" "$BASE/run_manifest.json" <<'PY'
import hashlib, json, pathlib, sys
root, path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
record = json.loads(path.read_text(encoding='utf-8'))
for name, expected in record['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'Source changed: ' + name
for name, expected in record['warm_start_sha256'].items():
    sha = hashlib.sha256()
    with open(name, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(chunk)
    assert sha.hexdigest() == expected, 'Warm-start checkpoint changed: ' + name
PY
    RUN="$BASE/$CURRENT_MODEL"
    mkdir -p "$RUN"
    echo "START $CURRENT_MODEL $(date --iso-8601=seconds)"
    printf 'running %s\n' "$CURRENT_MODEL" > "$BASE/status.txt"
    "$PYTHON" -u "$ROOT/tools/train.py" \
        "$ROOT/configs/experiments/reliability_fusion_v3/$CURRENT_MODEL.py" \
        --work-dir "$RUN" --cfg-options "randomness.seed=$SEED" \
        > "$RUN/console.log" 2>&1
    test -s "$RUN/epoch_6.pth"
    echo "DONE $CURRENT_MODEL $(date --iso-8601=seconds)"
done
printf 'completed\n' > "$BASE/status.txt"
echo "ALL_COMPLETED $BASE"
