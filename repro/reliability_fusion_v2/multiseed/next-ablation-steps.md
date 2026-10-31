# V2 多种子分析后的下一步实验

目标：保留已观察到的遮挡收益，定位模糊 EAA 行人退步。当前主模型、同增强控制组均已有 seed=0、1、2 的固定第 6 轮结果。

## 先做两项 seed=0 消融

| 配置 | 改动 | 保留设置 | 重点问题 |
|---|---|---|---|
| gate_aug_attention_only.py | 关闭两阶段 query 图像门控 | 注意力门控、图像增强、清晰约束、BN 策略 | query 入口是否贡献遮挡收益或模糊行人退步 |
| gate_aug_no_anchor.py | 清晰约束系数设为 0 | 四个门控、图像增强、BN 策略 | 清晰约束是否影响干净精度及退化适应 |

均从官方 exps/mssf_pp_vod/epoch_40.pth 开始，batch size 1、6 epoch、相同学习率与数据划分，seed=0。使用预先固定的 epoch_6.pth 完整评测，不按验证结果为各组另选 epoch。attention_only 的约束平均仅包含剩余注意力门控，这是移除 query 组件的整体消融；不等同于固定其他所有门控梯度的单参数试验。

## 已准备好的顺序运行脚本

脚本位于 /home/d514/桌面/zcy/MSSF_zcy/tools/analysis_tools/run_reliability_v2_ablation.sh。只完成了 Bash 语法和配置预检，本次分析未启动新训练。它会顺序完成两次训练及各自完整退化评测，复用已保存的 seed=0 control_aug 权重。遇到错误会停止；重复启动时明确提示任务锁占用。

在服务器终端执行：

```bash
conda activate mssf-bench
ROOT='/home/d514/桌面/zcy/MSSF_zcy'
BATCH="$ROOT/work_dirs/reliability_fusion_v2/ablation/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/work_dirs/reliability_fusion_v2/ablation"
nohup bash "$ROOT/tools/analysis_tools/run_reliability_v2_ablation.sh" "$BATCH" > "$BATCH.log" 2>&1 < /dev/null &
echo "批次目录：$BATCH"
echo "后台进程：$!"
tail -F "$BATCH.log"
```

整个队列出现 ALL_COMPLETED 才算结束。每个配置的训练日志位于批次下的 <variant>/console.log，评测结果为 <variant>/robustness/summary.json。结果也自动更新 V2 README 与独立 JSON 归档。自动诊断区只显示最近一次运行，三个 seed 的主实验汇总和原始档案会保留。运行期间保持分支、提交和源代码不变。

## 完成后如何判断

1. 将两个消融版本与同 seed 的完整 V2、同增强控制组一起比较全部四种图像条件，不只看一项最高值。
2. 优先检查遮挡 EAA/DC 的收益能否保留、模糊 EAA 行人 AP 是否恢复，以及干净集是否出现损失。
3. 如果关闭 query 后模糊改善，只能先形成 query 组件影响该权衡的假设；结合遮挡表现判断，不直接宣称原因已被证明。
4. 对有解释价值的消融补 seed=1、2，再按配对差值汇总均值和样本标准差。之后才据证据调整 V3 的门控或损失设计。

三种训练 seed 均来自同一官方预训练 checkpoint，所度量的是本次微调随机性的变化。评测固定种子为 0，便于各组比较相同合成条件。三个 seed 不是三个独立数据集，也不足以证明真实天气泛化或统计显著。
