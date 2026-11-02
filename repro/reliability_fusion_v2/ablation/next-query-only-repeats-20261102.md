# Query-only 完成后的下一步（2026-11-02）

Query-only 已于 2026-10-31 21:08 完成 seed=0 的 6 epoch 训练及完整退化评测。模糊 EAA 已接近同增强控制，但遮挡 EAA/DC 低于完整 V2。下一步只补 query-only 的 seed=1、2，复用两组既有 control_aug 和完整 V2 结果；不重复训练这些模型。

## 已准备的运行方式

启动脚本支持 RELIABILITY_SEEDS，未设置时保持原来的 seed=0 行为。每个 seed 自动使用与它配对的旧控制组 checkpoint。输出路径在项目 work_dirs 下，nas 链接指向 /mnt/nas-share/MSSF_zcy/experiments/reliability_fusion_v2/ablation。准备时系统盘约剩 4.69 GiB，NAS 约有 3481 GiB 可用，因此新一轮结果写到 NAS。已有实验没有迁移。

本次仅完成语法、两个 seed 的控制组映射和默认行为的 dry-run 预检，未启动复验。服务器终端执行：

```bash
conda activate mssf-bench
ROOT='/home/d514/桌面/zcy/MSSF_zcy'
BATCH="$ROOT/work_dirs/reliability_fusion_v2/ablation/nas/query_only_repeat_$(date +%Y%m%d_%H%M%S)"
nohup env RELIABILITY_SEEDS="1 2" bash "$ROOT/tools/analysis_tools/run_reliability_v2_ablation.sh" "$BATCH" gate_aug_query_only > "$BATCH.log" 2>&1 < /dev/null &
echo "批次目录：$BATCH"
tail -F "$BATCH.log"
```

顺序为 seed=1 query-only 训练和四条件完整评测，再执行 seed=2。每组 seed 的训练日志在 <批次>/seedN/gate_aug_query_only/console.log，评测为该目录下 robustness/summary.json。只有 ALL_COMPLETED 才表示整批完成。每次为 6 epoch、30834 步、seed 对应官方初始化；评测固定 seed=0 保证相同合成条件。每次保留两个最近 checkpoint，原训练和组件消融结果不覆盖。

## 复验后如何决定

1. 对三个 seed 分别计算 query-only 与匹配 control_aug、完整 V2 的差值，再汇总均值和样本标准差，不混用不同 seed 的控制组。
2. 查看模糊 EAA 行人能否稳定恢复，同时量化放弃了多少遮挡 EAA/DC 收益和干净精度。
3. 若这种权衡在三个 seed 重现，再针对注意力输出门控的作用阶段与抑制强度设计下一版；单种子结果不足以直接证明注意力门控是原因。
4. 组件比较同时移除了对应清晰约束项，并按剩余门控平均；要分离精确的前向作用与约束梯度，后续需控制每个门控的约束权重。
5. 完整 V2 保持主要工作候选；query-only 暂用于分析，不直接凭模糊一项更高替换它。
