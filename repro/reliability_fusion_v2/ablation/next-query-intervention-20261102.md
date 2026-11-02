# Attention 干预结束后的下一步（2026-11-02）

完整 V2 继续作为主候选。本轮固定权重 attention 干预没有恢复模糊精度，反而损失遮挡表现，不支持直接把两个 attention 门控设为恒等。[本轮结论](../attention_intervention/conclusion-seed0-20261102.json)；[完整 V2 README](../../../README_reliability_fusion_v2.md)。

## 1. 下一项只做 query 前向干预，不训练

固定同一个完整 V2 seed=0、第 6 轮 checkpoint；两个 attention 门控参数保持学习值，两个 query 门控按下列策略在内存中改为恒等。所有特征分支保留，无效投影掩码保留。

| 策略 | middle_encoder query | backbone query |
|---|---|---|
| learned | 学习值 | 学习值 |
| middle_identity | 有效投影乘数为 1 | 学习值 |
| backbone_identity | 学习值 | 有效投影乘数为 1 |
| both_identity | 有效投影乘数为 1 | 有效投影乘数为 1 |

“参数不变”不意味着网络后续特征或门控输出都不变：上游 query 干预可以改变下游 attention/query 的输入。该实验测量联合训练模型对干预的响应，不等同于重训练组件消融。

四策略 × clean/dark/blur/occlusion × 同一 1296 帧，共 16 组。按上一轮实际约 22 分钟，预计本轮约 22–27 分钟，受服务器负载影响。先完成这一次诊断，再决定是否需要新的算法训练，避免直接搜索大量门控强度。

在服务器执行一次：

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 PYTHONDONTWRITEBYTECODE=1
RUN="$PWD/work_dirs/reliability_fusion_v2/ablation/nas/query_intervention_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/analysis_tools/reliability_attention_intervention.py \
  --gate-kind query --samples 1296 --conditions clean dark blur occlusion \
  --work-dir "$RUN" --record-readme \
  > "$RUN/console.log" 2>&1 < /dev/null &
echo "进程：$!"
echo "结果目录：$RUN"
printf '%s\n' "$RUN" > work_dirs/reliability_fusion_v2/last_query_intervention_path.txt
```

脚本名为旧 attention 入口，现已增加 `--gate-kind query`；省略此参数仍默认 attention，旧命令含义保持不变。GPU 队列锁冲突会直接退出，启动前也应确认 GPU 上没有其他未使用本项目锁的任务。

## 2. 查看进度

```bash
cd '/home/d514/桌面/zcy/MSSF_zcy'
RUN="$(cat work_dirs/reliability_fusion_v2/last_query_intervention_path.txt)"
tail -n 25 "$RUN/console.log"
```

每完成一组会输出 `RESULT`；只有最后 `COMPLETED .../summary.json` 才代表全部 16 组完成。异常退出时保留旧目录，使用新目录重跑。

正式完成后结果自动写到 V2 README 的独立“固定权重 query 门控干预”区块，不覆盖 attention 的结果；原始 JSON 归档到 `repro/reliability_fusion_v2/query_intervention/<run_id>.json`。服务器 README 更新后，桌面快照需另行同步。

## 3. 完成后如何决定算法改动

- 先核对 learned 能否近似复现本轮 attention 实验中的完整 V2。
- 若某个 query 阶段恒等能恢复模糊 EAA 行人，同时 clean 和遮挡没有明显损失，下一步只提出该阶段一个弱化抑制候选，例如 `g_eff = 1 + alpha * (g - 1)`，固定一个预先说明的 alpha，再从同一起点按匹配协议训练验证。
- 若改善模糊始终损失遮挡，保留完整 V2，不把全局弱化 query 当成解决方案。
- 若 query 恒等也不恢复模糊，则这两轮推理诊断未支持直接削弱前向门控的方案；下一步转向匹配训练中共享特征漂移和正则项的检查，不再继续盲目扫描推理门控强度。
- 只有一个训练 seed 的干预结果用于定位。新的算法候选仍需与同增强控制按固定第 6 轮比较，先 seed=0 筛查，再补 seed=1、2；不能把这次的推理分数作为新训练模型的正式结果。

## 本轮文件与验证

| 文件 | 位置 | 修改或内容 |
|---|---|---|
| [干预脚本](../../../tools/analysis_tools/reliability_attention_intervention.py) | `apply_intervention()` | 根据 attention/query 选择目标，检查只改选中输出层参数，记录另一类门控参数未改 |
| 同上 | `evaluate()` / `main()` | 增加 `--gate-kind`，在报告中区分两类门控；默认行为仍为 attention |
| 同上 | `record_full_result()` | 两类结果分别归档、分别更新 README 区块；记录锁放在忽略的 work_dirs 下 |
| [Attention 原始结果](../attention_intervention/20261102_160250.json) | `results` | 16 组完整 AP、门控分布与权重不可变核验 |
| [Attention 结论](../attention_intervention/conclusion-seed0-20261102.json) | `rows` / `audit` | 配对差值、逐类分析、来源核验和下一步决策 |
| [新入口接入检查](query-intervention-smoke-20261102.json) | `runs` | 两种 gate-kind 各 4 帧 clean/blur、四策略；不是正式精度结果 |

本轮没有改变训练算法、训练配置或已有 checkpoint。完整 query 诊断尚未启动。
