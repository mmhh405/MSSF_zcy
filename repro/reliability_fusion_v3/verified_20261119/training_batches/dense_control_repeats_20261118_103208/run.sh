#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
PYTHON="$2"
BATCH="$3"
cd "$ROOT"
exec 9> "$ROOT/work_dirs/reliability_fusion_v3/.paired-training.lock"
flock -n 9 || { echo "FAILED another V3 queue owns the lock"; exit 1; }
CURRENT=preflight
trap 'code=$?; if (( code != 0 )); then echo "FAILED $CURRENT exit=$code"; fi' EXIT
"$PYTHON" - "$BATCH" <<'PY'
import hashlib,json,pathlib,subprocess,sys,torch
from mmengine.config import Config
root=pathlib.Path.cwd();base=pathlib.Path(sys.argv[1])
config=Config.fromfile('configs/experiments/reliability_fusion_v3/dense_control.py')
assert torch.cuda.is_available()
assert config.model.early_budget.mode=='dense'
assert config.train_cfg.max_epochs==6 and config.train_dataloader.batch_size==1
warm=(root/config.load_from).resolve()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
files=subprocess.check_output(['git','ls-files','-z','*.py']).split(b'\0')
record=dict(seeds=[1,2],epochs=6,batch_size=1,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initialization_checkpoint=str(warm),warm_start_sha256=digest(warm),source_sha256={p.decode():digest(root/p.decode()) for p in files if p})
(base/'run_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('PREFLIGHT_OK',sys.executable,flush=True)
PY
printf '%s\n' "$BATCH" > "$ROOT/work_dirs/reliability_fusion_v3/last_dense_control_repeats_path.txt"
printf '%s\n' "$$" > "$BATCH/queue.pid"
for SEED in 1 2; do
    CURRENT="seed$SEED"
    "$PYTHON" - "$BATCH/run_manifest.json" <<'PY'
import hashlib,json,pathlib,sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
for p,sha in m['source_sha256'].items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==sha,'Source changed: '+p
assert hashlib.sha256(pathlib.Path(m['initialization_checkpoint']).read_bytes()).hexdigest()==m['warm_start_sha256'],'Warm start changed'
PY
    RUN="$BATCH/seed$SEED"
    mkdir -p "$RUN"
    echo "START seed=$SEED $(date --iso-8601=seconds)"
    printf 'running seed=%s\n' "$SEED" > "$BATCH/status.txt"
    "$PYTHON" -u "$ROOT/tools/train.py" \
      "$ROOT/configs/experiments/reliability_fusion_v3/dense_control.py" \
      --work-dir "$RUN" \
      --cfg-options "randomness.seed=$SEED" \
      "reliability_experiment.variant=v3-dense-control-seed${SEED}" \
      > "$RUN/console.log" 2>&1
    "$PYTHON" - "$RUN" "$SEED" <<'PY'
import json,pathlib,sys,torch
run=pathlib.Path(sys.argv[1]);seed=int(sys.argv[2]);s=json.loads((run/'reliability_result.json').read_text())['summary']
assert s['training_completed'] and s['random_seed']==seed and s['validation_epoch']==6 and s['iterations']==30834 and s['samples']==1296
ck=torch.load(run/'epoch_6.pth',map_location='cpu')
assert ck['meta']['epoch']==6 and ck['meta']['iter']==30834
assert all(torch.isfinite(v).all().item() for v in ck['state_dict'].values() if v.is_floating_point())
print('RESULT',seed,s['eaa_map'],s['dc_map'],flush=True)
PY
    echo "DONE seed=$SEED $(date --iso-8601=seconds)"
done
printf 'completed\n' > "$BATCH/status.txt"
echo "ALL_COMPLETED $BATCH"
