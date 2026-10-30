# MSSF-PP 可靠性感知融合 V2

分支：`feat/reliability-gate-v2`，从 V1 全量退化分析提交 `8fde873` 创建。V1 的模型、配置和结果保留在原分支/标签及 `README_reliability_fusion.md` 中。

当前阶段：代码与接入验证完成，共新增 18820 个参数、4 个门控。正式实验使用顺序队列运行三个 6 epoch 任务；实时状态见队列 status.json，尚无 V2 完整验证集精度或鲁棒性结论。

## 设计与边界

V1 在图像退化时门控仍约为 0.99。V2 将特征权重改为 `g = 2 * sigmoid(z)`，最后一层权重和偏置置零，使有效投影上的初始 `g=1`。在相同 logit 导数尺度下，初始斜率为 0.5，而 V1 的 0.99 sigmoid 门控斜率约 0.0099。这只改善局部可学习性，不保证精度提升。

V2 具有两类独立门控，每个融合阶段均可单独启用：

- **query 图像入口**：在采样图像特征进入原 `query_proj` 混合投影前加权，保留雷达特征和位置输入。
- **注意力图像增量**：延续 V1 的位置，在图像增量投影后、原查询残差相加前加权。

门控输入为实际雷达/图像特征、可学习空间的一致性、投影几何，以及未归一化特征的 log-energy。权重范围为 0–2，可降低或放大特征，不能解释为已校准的置信概率。无效投影屏蔽图像项，但不等同独立雷达单模态模型。

训练时有 50% 概率对图像随机施加变暗、模糊或局部遮挡，另外 50% 为清晰样本。原始图像文件、雷达、标注和几何矩阵保持不变。训练扰动强度与位置随机，验证与测试默认仍使用原图；全量退化评测另行固定条件。

清晰样本约束为 `loss_gate_clean = λ * sum_valid_clean((g-1)^2) / N_valid_clean`，默认 λ=0.1。它**只限制门控在清晰样本上的漂移**，不是冻结教师的整网预测一致性，不保证其他可训练参数或预测完全不变。增强标记仅参与训练损失的样本选择，不进入门控预测网络。

V2 控制组与门控组均固定 BatchNorm/SyncBN 的运行统计量，仿射参数仍按原冻结策略决定是否训练；这是 V2 共同实验设置，因此应使用 V2 新控制组，不能直接把 V1 控制组视作完全匹配对照。图像分支沿用原参数冻结策略。默认体网络学习率 1e-5，新门控参数为其 10 倍；6 epoch、batch size 1、seed 0。以上是候选实验设置，并非已验证的最佳参数。

## 修改文件与入口

<!-- FILE_LOCATIONS_START -->

| 文件与入口行 | 修改内容 |
|---|---|
| [mmdet3d/models/layers/fusion_layers/reliability_gate_v2.py:10](mmdet3d/models/layers/fusion_layers/reliability_gate_v2.py#L10) | 居中权重、能量特征、训练用清晰样本约束缓存 |
| [mmdet3d/models/layers/fusion_layers/query_reliability.py:22](mmdet3d/models/layers/fusion_layers/query_reliability.py#L22) | 在原 query 混合投影之前门控图像项 |
| [mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py:360](mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py#L360) | 为 V2 透传清晰损失掩码，V1 保持原接口行为 |
| [mmdet3d/models/layers/fusion_layers/voxel_fusion_block.py:82](mmdet3d/models/layers/fusion_layers/voxel_fusion_block.py#L82) | 为注意力门控提供仅用于损失的清晰标记 |
| [mmdet3d/models/middle_encoders/pillar_scatter_fuse.py:143](mmdet3d/models/middle_encoders/pillar_scatter_fuse.py#L143) | 第一阶段 query 图像入口 |
| [mmdet3d/models/backbones/second_fuse.py:209](mmdet3d/models/backbones/second_fuse.py#L209) | 第二阶段 query 图像入口 |
| [mmdet3d/models/detectors/voxelnet_fuse.py:57](mmdet3d/models/detectors/voxelnet_fuse.py#L57) | 可选固定 BN 统计；loss 包装器汇总并消费清晰约束 |
| [mmdet3d/models/layers/fusion_layers/__init__.py:7](mmdet3d/models/layers/fusion_layers/__init__.py#L7) | 注册 V2 门控，保留 V1 类 |
| [mmdet3d/datasets/transforms/image_degradation.py:11](mmdet3d/datasets/transforms/image_degradation.py#L11) | 训练退化和含可靠性元信息的打包器 |
| [mmdet3d/datasets/transforms/__init__.py:4](mmdet3d/datasets/transforms/__init__.py#L4) | 注册 V2 数据变换 |
| [mmdet3d/engine/hooks/reliability_experiment_hook.py:28](mmdet3d/engine/hooks/reliability_experiment_hook.py#L28) | 支持独立 README/结果索引路径，默认 V1 路径不变 |
| [tools/analysis_tools/reliability_robustness.py:193](tools/analysis_tools/reliability_robustness.py#L193) | 支持自定义 V2 配置、checkpoint、README 与归档路径 |
| [tests/test_reliability_fusion_v2.py:26](tests/test_reliability_fusion_v2.py#L26) | 9 项 V2 行为、梯度、元信息隔离及消融配置检查 |
| [repro/reliability_fusion_v2/results.json:1](repro/reliability_fusion_v2/results.json#L1) | V2 独立的运行摘要索引 |
| [README.md:106](README.md#L106) | 项目入口链接 |
| [.gitignore:15](.gitignore#L15) | 忽略 V2 的临时结果锁文件 |
| [README_reliability_fusion_v2.md:1](README_reliability_fusion_v2.md#L1) | V2 功能、限制、配置、命令与结果记录 |
| [configs/experiments/reliability_fusion_v2/control_aug.py:2](configs/experiments/reliability_fusion_v2/control_aug.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/control_clean.py:2](configs/experiments/reliability_fusion_v2/control_clean.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/gate_attention_clean.py:2](configs/experiments/reliability_fusion_v2/gate_attention_clean.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/gate_aug_attention_only.py:2](configs/experiments/reliability_fusion_v2/gate_aug_attention_only.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/gate_aug_no_anchor.py:2](configs/experiments/reliability_fusion_v2/gate_aug_no_anchor.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/gate_both_clean.py:2](configs/experiments/reliability_fusion_v2/gate_both_clean.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/gate_main.py:2](configs/experiments/reliability_fusion_v2/gate_main.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/smoke_inference.py:2](configs/experiments/reliability_fusion_v2/smoke_inference.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [configs/experiments/reliability_fusion_v2/smoke_train.py:2](configs/experiments/reliability_fusion_v2/smoke_train.py#L2) | V2 控制、消融或接入检查配置；详下方配置表 |
| [repro/reliability_fusion_v2/verification.json:1](repro/reliability_fusion_v2/verification.json#L1) | 本次测试、梯度、BN 统计、源码哈希和真实接入记录 |
| [tools/analysis_tools/run_reliability_v2.py:1](tools/analysis_tools/run_reliability_v2.py#L1) | GPU 0 顺序运行三个正式实验，保存状态、日志与代码哈希，失败时停止后续实验 |

<!-- FILE_LOCATIONS_END -->

## 配置及消融

所有配置位于 `configs/experiments/reliability_fusion_v2/`：

| 配置 | 用途 |
|---|---|
| `control_clean.py` | 原结构、无图像退化，验证 V2 共同训练设置 |
| `control_aug.py` | 原结构＋图像退化，隔离增强本身的收益 |
| `gate_attention_clean.py` | 仅 V2 注意力门控包，无退化、无清晰约束（同时包含居中参数化与能量特征） |
| `gate_both_clean.py` | query＋注意力门控、清晰约束，无退化 |
| `gate_aug_attention_only.py` | 与主版本相同退化/约束，关闭 query 门控 |
| `gate_aug_no_anchor.py` | 与主版本相同退化/两类门控，关闭清晰约束 |
| `gate_main.py` | V2 主候选：两类门控＋退化训练＋清晰约束 |
| `smoke_inference.py` | 前 16 帧、未训练的初始恒等门控 |
| `smoke_train.py` | 四次优化器更新，仅接入验证 |

V2 从**官方 `exps/mssf_pp_vod/epoch_40.pth`** 初始化，不继承 V1 的近饱和门控权重。新增参数缺失是预期初始化；其他缺失或形状不匹配需要处理。正式训练之前固定代码提交。四步检查 checkpoint 不能用作正式模型。

## 接入检查

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
git switch feat/reliability-gate-v2
python tests/test_reliability_fusion_v2.py
```

16 帧推理：

```bash
RUN="work_dirs/reliability_fusion_v2/smoke_inference/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/test.py configs/experiments/reliability_fusion_v2/smoke_inference.py \
  --checkpoint exps/mssf_pp_vod/epoch_40.pth --metric VODMetric \
  --work-dir "$RUN" --pkl_save_dir "$RUN/predictions" 2>&1 | tee "$RUN/console.log"
```

四步反向传播：

```bash
RUN="work_dirs/reliability_fusion_v2/smoke_train/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/train.py configs/experiments/reliability_fusion_v2/smoke_train.py \
  --work-dir "$RUN" 2>&1 | tee "$RUN/console.log"
```

## 正式实验

先完成 `control_clean.py` 与 `control_aug.py`，再运行 `gate_main.py`；其他配置用于比较最简 V2 注意力门控包、query 入口、增强和清晰约束。当前未单独消融能量特征，因此不能把最简 V2 与旧版的差异仅归因于门控函数或初始化。各运行使用独立 `work_dirs/reliability_fusion_v2/<variant>/<timestamp>`。同一 GPU 顺序运行。

训练命令模板：

```bash
RUN="work_dirs/reliability_fusion_v2/gate_main/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
  configs/experiments/reliability_fusion_v2/gate_main.py \
  --work-dir "$RUN" > "$RUN/console.log" 2>&1 < /dev/null &
tail -f "$RUN/console.log"
```

主版本应与 `control_aug.py` 的同预算结果比较，并分别报告是否超过官方未额外微调基线。所有配置的训练/验证 Hook 均只更新本 V2 README 和 `repro/reliability_fusion_v2/results.json`，不会覆盖 V1 结果。完整逐轮历史在运行目录的 `reliability_result.json` 中；表中只显示每次运行最近一次验证值。


### 顺序队列

使用已激活的 mssf-bench 环境，在项目根目录执行：

```bash
QUEUE="work_dirs/reliability_fusion_v2/queue/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$QUEUE")"
nohup python -u tools/analysis_tools/run_reliability_v2.py --work-dir "$QUEUE" > "${QUEUE}.log" 2>&1 < /dev/null &
```

队列顺序为 control_clean、control_aug、gate_main，各 6 epoch、seed 0、batch size 1，均从官方 epoch_40.pth 初始化。队列的 status.json 记录进程、每组目录、代码提交与源文件哈希，失败时停止。每组 console.log 位于各自运行目录；SSH 断开后继续运行。代码在运行期间应保持不变。首组启动后，结果 Hook 自动更新 README/结果 JSON，所以后续组的全工作区 dirty 标记可能为真；队列仍逐组核对算法文件哈希与提交，确保代码一致。

## 训练后退化评测

沿用固定条件的诊断工具，显式给出 V2 配置、正式 checkpoint 和 V2 输出位置：

```bash
python tools/analysis_tools/reliability_robustness.py --samples 1296 \
  --conditions clean dark blur occlusion \
  --control-config configs/experiments/reliability_fusion_v2/control_aug.py \
  --gate-config configs/experiments/reliability_fusion_v2/gate_main.py \
  --control-checkpoint PATH_TO_CONTROL_EPOCH_6 \
  --gate-checkpoint PATH_TO_V2_EPOCH_6 \
  --readme README_reliability_fusion_v2.md \
  --archive-dir repro/reliability_fusion_v2/robustness \
  --work-dir UNIQUE_RUN_DIRECTORY
```

上述大写项需替换成真实训练产物路径。该脚本用于精度与门控响应，不用于测量 FPS；合成退化不代表真实夜间/天气。query 与注意力门控会分别统计。

## 本次验证记录

验收证据：[repro/reliability_fusion_v2/verification.json](repro/reliability_fusion_v2/verification.json)。实现代码提交为 `e237d7f`。

- 22 项测试通过：9 项 V2 测试、7 项 V1 回归、6 项退化诊断回归。
- 有效投影上的初始门控为 1；固定输入的注意力输出与冻结基线源码逐元素一致，query 图像入口初始也保持恒等。
- 16 帧真实推理完成，普通 EAA/DC 为 25.0645%/17.7489%，与原基线相同样本的接入检查一致。仅为小样本流程验证。
- 4 次优化器更新正常完成，四个门控的输出层权重均从零更新，243 个 BN 运行统计缓冲量与官方 checkpoint 逐项一致。
- 真实清晰批次的独立探针确认约束损失大于零，四个门控均收到非零梯度；同一批次改为非清晰标记后，该约束为零。探针只在丢弃的内存模型中临时设置偏置，没有覆盖任何 checkpoint。
- V2 的 4 帧 clean/blur 诊断接口通过，可分别记录 query 与注意力四个门控并写入独立 V2 归档。使用的是未训练、初始权重为 1 的 V2 检查 checkpoint，不是鲁棒性实验结论。
- 官方 checkpoint 与 V1 已训练 checkpoint 均仍可严格加载；V1 README、结果索引及全量结论文件与 `8fde873` 保持逐字节一致。

初始恒等门控和小样本训练只证明实现可用，不代表已经学到可靠性，也不能作为完整验证集成绩。四步检查的工作区 dirty 标记来自前一轮推理自动生成的 V2 结果文档；算法文件与记录中的实现提交一致。正式实验前应提交结果文档并确认工作区状态。

首先执行 V2 控制组，而非直接复用 V1 控制结果：`control_clean.py`、`control_aug.py`、`gate_main.py`，分别使用独立目录和相同预算。

## 训练与评测结果（自动更新）

<!-- RELIABILITY_RESULTS_START -->

| 实验 | 类型/状态 | 样本数 | 验证 epoch | EAA mAP (%) | DC mAP (%) | 代码 | 运行目录 |
|---|---|---:|---:|---:|---:|---|---|
| v2-untrained-inference | smoke_only | 16 | — | 25.0645 | 17.7489 | e237d7f1 | work_dirs/reliability_fusion_v2/smoke_inference/20261030_175948 |
| v2-four-iteration-backward | smoke_only | — | — | — | — | e237d7f1+dirty | work_dirs/reliability_fusion_v2/smoke_train/20261030_180638 |

<!-- RELIABILITY_RESULTS_END -->

## 受控图像退化诊断（自动更新）

<!-- ROBUSTNESS_RESULTS_START -->

子集诊断，不能与论文完整验证集分数比较。同一批 4 帧、官方原结构权重与未训练、初始权重为 1 的 V2 权重，仅验证接口，无训练。运行 `20261030_181240`；合成退化不等同真实夜间或恶劣天气。

| 条件 | 样本数 | 对照 EAA | 门控 EAA | 门控减对照 | 对照 DC | 门控 DC | 门控减对照 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 4 | 14.1414 | 14.1414 | +0.0000 | 7.5758 | 7.5758 | +0.0000 |
| blur | 4 | 2.5712 | 2.5712 | +0.0000 | 1.5152 | 1.5152 | +0.0000 |

| 条件 | 门控阶段 | 有效查询门控均值 | 标准差 | 低于 0.9 的比例 |
|---|---|---:|---:|---:|
| clean | backbone/attention | 1.0000000 | 0.0000000 | 0.00000 |
| clean | backbone/query | 1.0000000 | 0.0000000 | 0.00000 |
| clean | middle_encoder/attention | 1.0000000 | 0.0000000 | 0.00000 |
| clean | middle_encoder/query | 1.0000000 | 0.0000000 | 0.00000 |
| blur | backbone/attention | 1.0000000 | 0.0000000 | 0.00000 |
| blur | backbone/query | 1.0000000 | 0.0000000 | 0.00000 |
| blur | middle_encoder/attention | 1.0000000 | 0.0000000 | 0.00000 |
| blur | middle_encoder/query | 1.0000000 | 0.0000000 | 0.00000 |

[完整参数、逐类指标与门控分布](repro/reliability_fusion_v2/robustness/20261030_181240.json)

<!-- ROBUSTNESS_RESULTS_END -->
