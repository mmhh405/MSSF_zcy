# V3 新服务器启动修正

日期为 2026-11-04。服务器为 `192.168.1.144`，项目为 `/home/d514/zcy/MSSF_zcy`。旧服务器 `/home/d514/桌面/zcy/MSSF_zcy` 的路径不适用于此工作区。

本次失败发生在导入 `mmengine` 时，尚未进入训练。项目自带 `.runtime/mssf-bench/bin/python` 已包含 mmengine 0.8.4，修正不需要重装环境。

## 启动环境

```bash
source /home/d514/zcy/MSSF_zcy/activate.sh
python -c 'import sys, mmengine; print(sys.executable, mmengine.__version__)'
```

激活脚本由自身位置确定项目路径，将项目内环境加入 PATH，并设置 `MSSF_PYTHON` 和 `PYTHONPATH`。启动队列还会直接调用绝对路径的 Python，后台子进程不会依赖终端提示符或 shell 的 python 缓存。

## 配对队列

以下用于新建一次实验，已有队列运行时只查看进度。

```bash
source /home/d514/zcy/MSSF_zcy/activate.sh
BASE="$PWD/work_dirs/reliability_fusion_v3/paired_$(date +%Y%m%d_%H%M%S)_seed0"
mkdir -p "$BASE"
nohup bash tools/analysis_tools/run_reliability_v3_pair.sh \
  --work-dir "$BASE" --seed 0 > "$BASE/queue.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$BASE/queue.pid"
echo "$BASE"
```

队列先检查环境、CUDA、数据和初始化权重，再顺序训练 dense_control、gate_main，各六轮，batch size=1，seed=0。两个模型使用同一 V2 权重。为保持与 V2 的配对协议，不采用迁移测试中的 batch size=8。

`run_manifest.json` 记录解释器、依赖版本、设备、Git 状态、全部已跟踪 Python 源文件的 SHA-256 和热启动权重 SHA-256。每组训练前重新核对源代码及权重。运行期间如需开发新版本，应使用另一份工作树，以免改变本次实验。

## 查看进度

```bash
source /home/d514/zcy/MSSF_zcy/activate.sh
BASE="$(cat work_dirs/reliability_fusion_v3/last_paired_path.txt)"
tail -n 20 "$BASE/queue.log"
if [ -f "$BASE/gate_main/console.log" ]; then
  tail -n 20 "$BASE/gate_main/console.log"
else
  tail -n 20 "$BASE/dense_control/console.log"
fi
```

出现 `ALL_COMPLETED` 表示两组都正常结束。若出现 `FAILED`，检查对应组的 console.log。旧的失败目录保留用于诊断，不会被覆盖。
