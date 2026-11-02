# Query-only 三 seed 之后的实验步骤（2026-11-02）

当前主候选保留完整 V2。query-only 的模糊 EAA 相对各自控制组三次均为正，但遮挡 EAA/DC 三次均低于完整 V2；模糊 DC 也三次低于完整 V2。完整比较见 [三 seed 汇总](query-only-multiseed-summary-20261102.json) 和 [V2 README](../../../README_reliability_fusion_v2.md)。

## 本轮新增文件与功能

| 文件 | 位置 | 功能 |
|---|---|---|
| [推理干预脚本](../../../tools/analysis_tools/reliability_attention_intervention.py) | `apply_intervention()` | 严格加载同一完整 V2 checkpoint，按策略只将选中 attention 门控的输出层权重和偏置清零；query 权重及其他参数保持加载值 |
| 同上 | `evaluate()` | 同帧、同退化、同评测随机种子，核验原始帧索引，记录逐类 AP、门控分布、权重哈希及运行来源 |
| 同上 | `record_full_result()` | 完整评测结束后归档 JSON，并更新 V2 README 的独立诊断区块 |
| 同上 | `main()` | 参数入口与 GPU 队列锁，拒绝覆盖已有结果 |
| [三 seed 汇总](query-only-multiseed-summary-20261102.json) | `rows` / `per_class` / `audit` | 保存三个 seed 配对均值、样本 SD、逐类结果和证据核验 |

脚本复用 [原退化工具](../../../tools/analysis_tools/reliability_robustness.py) 的图像退化、帧索引恢复和门控观察逻辑。本轮没有修改训练算法、配置或已有 checkpoint。

## 1. 固定同一完整 V2 权重，做四策略诊断

两个 query 门控保持学习值，attention 特征分支始终保留。只改变注意力输出门控的前向乘数：

| 策略 | middle_encoder attention | backbone attention |
|---|---|---|
| learned | 学习值 | 学习值 |
| middle_identity | 有效投影乘数固定为 1 | 学习值 |
| backbone_identity | 学习值 | 有效投影乘数固定为 1 |
| both_identity | 有效投影乘数固定为 1 | 有效投影乘数固定为 1 |

实现利用 V2 的 `g=2*sigmoid(z)`，只在内存中令选中的最终 logit 层为零。投影无效时仍保留硬掩码；原文件不改写。query 参数相同，但上游特征变化后，后续 query 的输入和输出也可能随之变化，因此结果是整个网络对这一干预的响应。

固定 seed=0 的完整 V2 第 6 轮权重，四策略 × clean/dark/blur/occlusion × 同一 1296 帧，共 16 组评测。无训练、无超参数搜索。大致耗时取决于服务器负载，以日志进度为准。

在已连接服务器的终端执行一次：

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1
RUN="$PWD/work_dirs/reliability_fusion_v2/ablation/nas/attention_intervention_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/analysis_tools/reliability_attention_intervention.py \
  --samples 1296 --conditions clean dark blur occlusion \
  --work-dir "$RUN" --record-readme \
  > "$RUN/console.log" 2>&1 < /dev/null &
echo "进程：$!"
echo "结果目录：$RUN"
printf '%s\n' "$RUN" > work_dirs/reliability_fusion_v2/last_attention_intervention_path.txt
```

该命令在 `feat/reliability-gate-v2` 的现有项目中执行，输出写到 NAS。若 GPU 队列锁被占用，脚本直接报错退出，不会等待或自动重试。启动前查看 GPU 是否有其他不使用本项目锁的任务。

## 2. 查看进度与确认完成

```bash
cd '/home/d514/桌面/zcy/MSSF_zcy'
RUN="$(cat work_dirs/reliability_fusion_v2/last_attention_intervention_path.txt)"
tail -n 25 "$RUN/console.log"
```

每完成一组会输出 `RESULT <策略> <条件> <EAA> <DC>`；16 组全部完成后才输出 `COMPLETED .../summary.json`。`progress.json` 可查看已完成组数。异常堆栈或进程退出不代表评测完成；失败后使用新目录重跑，原结果保留。

```bash
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
source = p / ('summary.json' if (p / 'summary.json').exists() else 'progress.json')
r = json.loads(source.read_text())
print('完成:', r['valid'], '组数:', len(r['results']), '/16')
for row in r['results']:
    print(row['policy'], row['condition'], row['eaa_map'], row['dc_map'])
PY
```

正式完成后，脚本会自动在 [V2 README](../../../README_reliability_fusion_v2.md) 的“固定权重注意力门控干预”一节写结果；完整 JSON 同时保存在 `repro/reliability_fusion_v2/attention_intervention/<run_id>.json`。这些文件更新在服务器，本机桌面副本需随后同步。

## 3. 结果如何决定后续训练

- 先核对 learned 策略能否近似复现原 seed=0 完整 V2 四条件结果，再解释干预差值。明显不一致时先查环境、配置、权重与帧对齐。
- 重点同时查看模糊 EAA 的行人 AP、模糊 DC、遮挡 EAA/DC 和 clean。不能仅凭一个 mAP 选择版本。
- 如果单个阶段设为恒等可改善模糊，且遮挡基本保留，再提出该阶段较弱抑制或残差混合的一个明确候选，并用匹配控制重新训练验证。
- 若模糊恢复始终伴随遮挡损失，应保留完整 V2，继续检查退化特征及空间目标分布，而不是直接用 query-only 替换。
- 这里仅是一个训练 seed 的固定权重诊断，包含联合训练后的特征适应；它不等同于重新训练组件消融，也不足以确定新的模型优越性。后续候选仍需固定协议、多 seed 验证。

## 接入验证

已用均匀选取的 4 帧在 clean/blur 下运行全部四种策略（8 组）；严格加载通过、原始帧索引匹配、选中有效门控恒为 1、query 参数未改、原 checkpoint SHA-256 未变。小样本 AP 不作为正式精度结果。[验证记录](attention-intervention-smoke-20261102.json)。

截至本说明交付时，完整 1296 帧诊断尚未启动，也没有启动新的训练。
