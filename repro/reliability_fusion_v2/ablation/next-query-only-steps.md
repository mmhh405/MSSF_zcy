> 进度更新：本计划的 seed=0 query-only 已完成。下一步见 [两个种子的复验](next-query-only-repeats-20261102.md)。

# 两项消融完成后的下一步

已有结果：seed=0 的 attention_only 和 no_anchor 均完成 6 epoch 和四种条件各 1296 帧评测。关闭 query 没有解决模糊问题，且遮挡 DC 低于完整 V2 1.6646 个百分点；关闭清晰约束差异较小，不能据一个 seed 决定删除。

下一步先补 gate_aug_query_only：只保留两个 query 门控，关闭两个注意力输出门控。与 control_aug（无门控）、attention_only、完整 V2（两类门控）一起形成组件组合对比。query_only 仍使用图像增强、清晰约束系数 0.1、相同 BN 策略、seed=0、batch=1、6 epoch，从官方权重开始；约束平均仅包含剩余 query 门控，因此是组件级消融。

已提供新配置及可选择配置的启动脚本。此次只准备配置和接入检查，未启动 query_only 正式训练。服务器终端执行：

```bash
conda activate mssf-bench
ROOT='/home/d514/桌面/zcy/MSSF_zcy'
BATCH="$ROOT/work_dirs/reliability_fusion_v2/ablation/query_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/work_dirs/reliability_fusion_v2/ablation"
nohup bash "$ROOT/tools/analysis_tools/run_reliability_v2_ablation.sh" "$BATCH" gate_aug_query_only > "$BATCH.log" 2>&1 < /dev/null &
echo "批次目录：$BATCH"
tail -F "$BATCH.log"
```

该命令只运行新的 query_only，随后复用原 seed=0 同增强控制权重进行完整退化评测；不会重跑已完成的两项消融。出现 ALL_COMPLETED 才算整批完成。训练日志位于批次的 gate_aug_query_only/console.log，评测汇总为该目录的 robustness/summary.json。

判断重点：模糊 EAA 行人是否恢复、遮挡 EAA/DC 收益保留多少，以及干净精度是否改变。该实验用于定位注意力输出门控的贡献；即使出现改善，也应保留单 seed 的限制，随后对关键组件配置补 seed=1、2。现有完整 V2 保持为工作候选，不依据本轮结果直接移除 query 或清晰约束，也不据此宣称模糊原因已被证明。
