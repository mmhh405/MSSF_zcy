# Query 干预后的复验步骤（2026-11-02）

当前完整 V2 保留为主候选。[本轮 query 诊断结论](../query_intervention/conclusion-seed0-20261102.json)；[完整 README](../../../README_reliability_fusion_v2.md)。

## 本轮得到的候选线索

仅 backbone query 恒等相对同一次完整 V2：模糊 EAA/DC +0.0926/+0.1012，clean +0.0220/+0.0129，dark +0.0298/-0.0172，遮挡 +0.0032/-0.0078，单位为百分点。增益很小，不能宣称显著或所有类别受益。模糊 EAA 行人 +0.1374，但模糊 DC 行人 -0.1217；模糊 EAA 整体也仍低于已归档的同增强控制组。

middle query 恒等或两个 query 都恒等均未改善模糊，并明显损失遮挡 DC。不继续扩大这些策略的种子。

## 1. 下一步只复验已有两个权重，不训练

复用完整 V2 seed=1、2 的固定第 6 轮 checkpoint，每个 seed 仅比较 learned 与 backbone_identity 两策略，两个 attention 参数和 middle query 参数保持加载值。四条件、同一 1296 帧；每个 seed 8 组，共 16 组，预计约 22–27 分钟。

已经准备 [两 seed 顺序启动器](../../../tools/analysis_tools/run_query_backbone_repeats.py)。启动器会从原始训练归档核验 checkpoint SHA-256，记录代码哈希，在两组之间检查代码未变，并拒绝覆盖旧输出。每个子任务使用项目 GPU 队列锁，锁冲突会失败退出。启动前也应确认没有其他未使用本项目锁的 GPU 任务。

在服务器终端执行一次：

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1
BATCH="$PWD/work_dirs/reliability_fusion_v2/ablation/nas/query_backbone_repeats_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH"
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/analysis_tools/run_query_backbone_repeats.py \
  --work-dir "$BATCH" > "$BATCH/queue.log" 2>&1 < /dev/null &
echo "进程：$!"
echo "结果目录：$BATCH"
printf '%s\n' "$BATCH" > work_dirs/reliability_fusion_v2/last_query_backbone_repeat_path.txt
```

启动器只运行评测，不会调用 train.py，也不会写回 checkpoint。

## 2. 查看进度与完成条件

```bash
cd '/home/d514/桌面/zcy/MSSF_zcy'
BATCH="$(cat work_dirs/reliability_fusion_v2/last_query_backbone_repeat_path.txt)"
tail -n 20 "$BATCH/queue.log"
```

先后出现 START 1、DONE 1、START 2、DONE 2，最后 ALL_COMPLETED 才代表两个 seed 的评测全部结束。每个 seed 的详细进度在其 console.log：

```bash
tail -n 20 "$BATCH/seed1/console.log"
tail -n 20 "$BATCH/seed2/console.log"
```

结果分别是 seed1/summary.json、seed2/summary.json；批次状态在 batch.json。若失败，先查看队列及对应 seed 的日志；重跑使用新目录，以保留已有记录。

每个子任务会在 `repro/reliability_fusion_v2/query_intervention/<run_id>.json` 归档完整结果，并将训练 seed 与评测 seed 分开记录。README 的自动 query 区块显示最近一次子任务；本轮 seed=0 结论和原始归档保留，随后应将三个 seed 做配对汇总，不将某个单 seed 的自动表当成三 seed 结论。

## 3. 复验结束后的决策

- 对每个训练 seed，用本轮 backbone_identity 减同轮 learned，再汇总三个配对差值的均值、样本 SD、逐 seed 方向和逐类 AP；使用普通 3D AP 的 EAA/DC。
- 保持当前已固定的策略、四条件和第 6 轮权重，不追加强度扫描或挑选最佳 epoch。
- 如果模糊 EAA/DC 的改善能在其余 seed 重复，且没有明显 clean/遮挡损失，再提出一个训练候选：仅对 backbone query 使用固定 alpha=0.5 的残差混合 `g_eff=1+alpha*(g-1)`，保留另外三个门控。再从相同官方预训练权重、匹配增强协议训练 6 轮，先做 seed=0 筛查。
- 如果收益方向不能重复，或行人/遮挡出现明显代价，停止弱化 backbone query 这一候选，保留完整 V2，转向共享特征变化和清晰约束的训练诊断。
- 推理干预的数值不是新训练模型的结果。即使三个 seed 一致，也只覆盖这组微调随机性与开发验证集上的合成退化，不能据此宣称统计显著或真实天气泛化。

## 文件位置与接入检查

| 文件 | 位置 | 功能或改动 |
|---|---|---|
| [干预入口](../../../tools/analysis_tools/reliability_attention_intervention.py) | `main()` / `evaluate()` | 增加 --policies，自动保留 learned 参照；默认仍评测原四策略。--training-seed 只用于来源记录，评测 RNG 固定为 0 |
| 同上 | `record_full_result()` | README 分别标注训练 seed 和评测 seed |
| [复验启动器](../../../tools/analysis_tools/run_query_backbone_repeats.py) | `main()` | 核验现有 seed=1、2 权重，依次运行两个策略，记录批次状态及源文件哈希；支持 --dry-run |
| [接入验证](query-backbone-repeat-smoke-20261102.json) | `smoke` / `dry_run` | 两策略四帧 clean/blur，与之前相同策略的小样本输出一致；权重哈希和帧索引核验通过。dry-run 不启动完整评测 |

本轮没有改变模型训练算法、配置或已有 checkpoint。两 seed 完整复验尚未启动。
