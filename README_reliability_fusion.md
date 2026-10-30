# MSSF-PP 可靠性感知融合：第一版

分支：`feat/reliability-gate`。固定基线：`baseline/mssf-pp-vod-v1`（`a67da65`）。

当前阶段：首轮原模型对照与门控版本的 6 epoch 微调均已完成。两个门控共新增 5890 个参数。本轮未显示有说服力的精度提升；完整比较与门控诊断见文末。

## 实现的功能

- 在 MSSF-PP 的 `PointPillarsScatterFuse` 与 `SECONDFuse` 两个 MSDFF 融合阶段加入独立的逐体素门控。
- 使用雷达上下文特征、图像注意力增量、可学习特征空间中的余弦一致性及归一化投影位置/深度预测标量权重。
- 校验正深度、有限坐标和未填充图像范围。投影无效时将图像注意力增量置零；原查询/雷达残差继续保留。
- 在 `output_proj` 之后、原注意力残差相加之前门控：`output = identity + dropout(gate * image_update)`。代码中的 `identity` 是原查询，查询本身可能已包含图像信息；这并不是把全部图像信息从网络中清除，也不是雷达单模态回退。
- 默认配置不启用门控，原模型参数键与旧权重保持兼容。启用时有效投影的初始门控约为 0.99，接近原模型但并不完全等价；新增参数必须微调。
- 训练验证及独立评测完成后，将真实结果自动写入本 README，并保存 JSON 摘要和每个 epoch 的指标历史。

门控分数是学习到的特征权重，不是已经校准的置信概率。第一版不增加新的监督损失或模态退化增强；先验证门控本身，再单独消融增强。这是研究候选实现，尚不能据此声称具有学术新颖性或精度提升。

## 修改文件及位置

<!-- FILE_LOCATIONS_START -->

| 文件及入口行 | 修改内容 |
|---|---|
| [mmdet3d/models/layers/fusion_layers/reliability_gate.py:12](mmdet3d/models/layers/fusion_layers/reliability_gate.py#L12) | 新增门控、一致性特征、硬有效性屏蔽与梯度路径 |
| [mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py:41](mmdet3d/models/layers/fusion_layers/radar_img_cross_att.py#L41) | 透传开关；在图像增量投影后、原残差之前门控 |
| [mmdet3d/models/layers/fusion_layers/voxel_fusion_block.py:27](mmdet3d/models/layers/fusion_layers/voxel_fusion_block.py#L27) | 按增强后的正深度、真实图像范围构造投影上下文 |
| [mmdet3d/models/middle_encoders/pillar_scatter_fuse.py:86](mmdet3d/models/middle_encoders/pillar_scatter_fuse.py#L86) | 第一阶段启用门控并传递图像元信息 |
| [mmdet3d/models/backbones/second_fuse.py:152](mmdet3d/models/backbones/second_fuse.py#L152) | 第二阶段启用门控并传递图像元信息 |
| [mmdet3d/models/layers/fusion_layers/__init__.py:6](mmdet3d/models/layers/fusion_layers/__init__.py#L6) | 注册新门控模块 |
| [mmdet3d/engine/hooks/reliability_experiment_hook.py:28](mmdet3d/engine/hooks/reliability_experiment_hook.py#L28) | 原子更新 README/JSON；保存验证历史、代码状态及训练完成状态 |
| [mmdet3d/engine/hooks/__init__.py:6](mmdet3d/engine/hooks/__init__.py#L6) | 注册结果记录 Hook |
| [tools/train.py:6](tools/train.py#L6) | 训练入口尊重调用者选择的 GPU |
| [configs/experiments/reliability_fusion/mssf_pp_vod_control.py:2](configs/experiments/reliability_fusion/mssf_pp_vod_control.py#L2) | 同预算的原模型微调对照 |
| [configs/experiments/reliability_fusion/mssf_pp_vod_gate.py:2](configs/experiments/reliability_fusion/mssf_pp_vod_gate.py#L2) | 启用两个阶段门控的微调配置 |
| [configs/experiments/reliability_fusion/mssf_pp_vod_gate_smoke.py:2](configs/experiments/reliability_fusion/mssf_pp_vod_gate_smoke.py#L2) | 16 帧未训练推理检查 |
| [configs/experiments/reliability_fusion/mssf_pp_vod_gate_train_smoke.py:2](configs/experiments/reliability_fusion/mssf_pp_vod_gate_train_smoke.py#L2) | 两次优化器更新的训练接入检查 |
| [tests/test_reliability_fusion.py:24](tests/test_reliability_fusion.py#L24) | 7 项针对性测试，含冻结基线源码对照和结果记录集成 |
| [repro/reliability_fusion/results.json:1](repro/reliability_fusion/results.json#L1) | 每个实验最近一次实际结果的索引 |
| [repro/reliability_fusion/verification.json:1](repro/reliability_fusion/verification.json#L1) | 本次验收、数值回归范围和源码哈希 |
| [.gitignore:12](.gitignore#L12) | 忽略结果写入锁文件 |
| [README.md:102](README.md#L102) | 新增实验说明入口 |
| [README_reliability_fusion.md:1](README_reliability_fusion.md#L1) | 本版本功能、文件位置、运行命令、验收与结果汇总 |

<!-- FILE_LOCATIONS_END -->

## 配置与兼容性

`model.middle_encoder.optional_cfg.reliability_gate` 和 `model.backbone.optional_cfg.reliability_gate` 分别控制两个阶段：

```python
dict(enabled=True, hidden_channels=16, init_probability=0.99)
```

不设置该字段或设置 `enabled=False` 即不创建门控参数。原始 checkpoint 在启用配置中会缺少新增 `reliability_gate.*` 参数，这是预期的初始化行为；任何其他缺失键、额外键或形状不匹配都应排查。禁止把 2 步反向传播检查的 checkpoint 当作训练完成的模型。

当前微调起点：官方 `exps/mssf_pp_vod/epoch_40.pth`；原图像分支沿用冻结策略；batch size 1，AdamW 初始学习率 1e-5，6 个 epoch，随机种子 0，每 epoch 验证一次。原模型对照使用同一预算。以上为第一版实验设置，尚未证明最优。

## 验收与运行命令

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
set -o pipefail
python tests/test_reliability_fusion.py
```

16 帧未训练门控检查：

```bash
RUN="work_dirs/reliability_fusion/gate_smoke/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/test.py \
  configs/experiments/reliability_fusion/mssf_pp_vod_gate_smoke.py \
  --checkpoint exps/mssf_pp_vod/epoch_40.pth --metric VODMetric \
  --work-dir "$RUN" --pkl_save_dir "$RUN/predictions" 2>&1 | tee "$RUN/console.log"
```

2 个优化器更新的训练接入检查（不是正式微调）：

```bash
RUN="work_dirs/reliability_fusion/train_smoke/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/train.py \
  configs/experiments/reliability_fusion/mssf_pp_vod_gate_train_smoke.py \
  --work-dir "$RUN" 2>&1 | tee "$RUN/console.log"
```

正式实验前提交源码和配置，保持运行期间代码不变。分别运行同预算对照和门控版本：

```bash
RUN="work_dirs/reliability_fusion/control/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/train.py \
  configs/experiments/reliability_fusion/mssf_pp_vod_control.py \
  --work-dir "$RUN" 2>&1 | tee "$RUN/console.log"

RUN="work_dirs/reliability_fusion/gate/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python -u tools/train.py \
  configs/experiments/reliability_fusion/mssf_pp_vod_gate.py \
  --work-dir "$RUN" 2>&1 | tee "$RUN/console.log"
```

最终 checkpoint 独立评测使用 `mssf_pp_vod_gate.py` 与本次训练权重，显式设置新的 `--work-dir` 和 `--pkl_save_dir`。从官方权重开始新实验使用 `load_from`，不使用 `--resume` 恢复作者的优化器状态；自己的中断训练才使用 `--resume`。

## 验收记录

已完成以下实际检查，日志与证据见 `repro/reliability_fusion/verification.json`：

- 7 项测试通过，覆盖非法投影、图像填充区、非有限值、可训练梯度、配置隔离、原版注意力对照和 README 记录。
- 关闭门控后可严格加载官方 checkpoint；固定输入下，与冻结基线源码的注意力输出在 CPU 上逐元素一致。
- 独立 GPU 运行的前 16 帧与历史基线类别及普通 AP 一致，未达到逐位一致：最大分数差约 8e-5、图像框坐标差约 0.020 像素、朝向差约 2.23e-4 弧度。这里只报告观测差异，没有单独定位其来源。
- 启用门控完成 16 帧真实推理并自动写入结果，新增缺失键仅为预期的 24 个门控状态键。
- 完成 2 个优化器更新，loss/梯度有限、checkpoint 保存成功，两个阶段门控的输出层权重均从零发生实际更新。
- 完成另外一个 2 帧训练、16 帧验证的单 epoch 接入检查；验证指标、checkpoint 路径及训练完成状态均自动进入结果记录。

上述小实验均是接入检查；表中的小样本数值不能与论文完整验证集精度比较。正式 6 epoch 结果现已完成，见下方比较；接入检查数值仍不能当作正式精度。

## 训练与评测结果（自动更新）

Hook 按运行目录和时间标识更新每次运行的最近一次验证结果，不自动选择最好 epoch；完整历史保存在该运行目录的 `reliability_result.json`。`smoke_only` 行仅用于流程检查。缺失指标显示“—”。正式训练结束后状态更新为 `training_completed`，保留最近一次验证指标。

<!-- RELIABILITY_RESULTS_START -->

| 实验 | 类型/状态 | 样本数 | 验证 epoch | EAA mAP (%) | DC mAP (%) | 代码 | 运行目录 |
|---|---|---:|---:|---:|---:|---|---|
| 官方权重基线 | official_pretrained | 1296 | — | 63.2253 | 79.7823 | a67da656 | work_dirs/baseline_mssf_pp_vod/20261030_095709 |
| gate-untrained-inference | smoke_only | 16 | — | 24.9956 | 17.7489 | a67da656+dirty | work_dirs/reliability_fusion/gate_smoke/20261030_105333 |
| gate-two-iteration-backward | smoke_only | — | — | — | — | a67da656+dirty | work_dirs/reliability_fusion/train_smoke/20261030_105443 |
| gate-epoch-validation-integration | smoke_only | 16 | 1 | 25.4473 | 17.8030 | a67da656+dirty | work_dirs/reliability_fusion/epoch_smoke/20261030_105628 |
| control-finetune | training_completed | 1296 | 6 | 62.8621 | 79.6676 | d5d1a959 | work_dirs/reliability_fusion/control/20261030_111629 |
| reliability-gate | training_completed | 1296 | 6 | 62.8654 | 79.6756 | 7d67f8ea | work_dirs/reliability_fusion/gate/20261030_152139 |

<!-- RELIABILITY_RESULTS_END -->

结果索引：`repro/reliability_fusion/results.json`。README 和小体积结果索引可提交到 Git；数据、模型权重、优化器状态和完整预测继续存放在已忽略的实验目录/NAS。记录中的 `git_dirty_at_start` 表示启动时是否存在未提交修改，正式对照前应确认源码和配置已固定。

## 对照实验分析（2026-10-30）

运行目录：`work_dirs/reliability_fusion/control/20261030_111629`。训练正常完成：5139 帧训练、6 epoch、30834 次迭代；每次验证 1296 帧，未启用可靠性门控。启动时源码干净，代码提交为 `d5d1a959`。

| Epoch | EAA mAP (%) | DC mAP (%) |
|---:|---:|---:|
| 1 | 62.9696 | 79.6965 |
| 2 | 62.9720 | 79.6907 |
| 3 | 62.8650 | 79.3789 |
| 4 | 62.8975 | 79.6645 |
| 5 | 62.9251 | 79.7046 |
| 6 | 62.8621 | 79.6676 |

**首轮按预先约定使用第 6 个 epoch：EAA 62.8621%、DC 79.6676%。** 相对官方 checkpoint 的本机基线（63.2253% / 79.7823%），分别下降 0.3632 和 0.1147 个百分点。6 个 epoch 的两项指标均低于该初始基线，说明当前额外微调方案没有提高原模型精度。具体回落原因尚未单独定位，不据此判定过拟合。

接下来保持相同初始化、训练预算、学习率、batch size 和随机种子评估门控版本。门控实验仍从官方 `exps/mssf_pp_vod/epoch_40.pth` 初始化，不从本次对照微调的 `epoch_6.pth` 继续训练。

门控相对 **62.8621% / 79.6676%** 的变化用于评估同预算下的模块作用；相对 **63.2253% / 79.7823%** 的变化用于评估是否超过原官方权重基线。两种比较均需报告，不能仅凭超过退化后的对照就宣称已超过原始 MSSF。

本次使用普通 3D AP；截图上方的 R40 和 strict R40 另属不同口径。checkpoint 哈希、实验设置和完整 epoch 摘要见 [对照实验记录](repro/reliability_fusion/control-20261030_111629.json)。

## 第一版门控实验结论（2026-10-30）

两组均以官方 `epoch_40.pth` 初始化，batch size 1、初始学习率 1e-5、随机种子 0、6 epoch、30834 次更新；每轮验证 1296 帧。已核对保存的训练配置一致，代码提交差异仅为实验记录，门控模型 checkpoint 可严格加载。

| 模型 | 第 6 epoch EAA mAP (%) | 第 6 epoch DC mAP (%) |
|---|---:|---:|
| 官方 checkpoint（未额外微调） | 63.2253 | 79.7823 |
| 同预算原模型对照 | 62.8621 | 79.6676 |
| 第一版可靠性门控 | 62.8654 | 79.6756 |

**相对对照仅增加 0.0033 / 0.0080 个百分点；相对原官方 checkpoint 仍下降 0.3599 / 0.1067 个百分点。** 这轮单随机种子实验未显示有说服力的收益，不能宣称已改进原始 MSSF，也不能据单次实验作统计显著性结论。这里比较普通 3D AP，而非 R40、strict 或 BEV。

| Epoch | 对照 EAA | 门控 EAA | 对照 DC | 门控 DC |
|---:|---:|---:|---:|---:|
| 1 | 62.9696 | 62.9450 | 79.6965 | 79.6817 |
| 2 | 62.9720 | 62.9759 | 79.6907 | 79.6812 |
| 3 | 62.8650 | 62.8523 | 79.3789 | 79.4026 |
| 4 | 62.8975 | 62.9140 | 79.6645 | 79.6923 |
| 5 | 62.9251 | 62.9250 | 79.7046 | 79.7254 |
| 6 | 62.8621 | 62.8654 | 79.6676 | 79.6756 |

### 门控诊断：均匀抽取 32 帧

使用训练后的第 6 epoch checkpoint，沿完整验证集均匀选取 32 帧进行只读推理，直接记录有效投影体素的实际门控输出：

| 阶段 | 有效查询数 | 门控均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|
| 骨干融合 | 18256 | 0.9905014 | 0.0001425 | 0.9900860 | 0.9910166 |
| 中间编码器 | 20021 | 0.9902124 | 0.0000578 | 0.9900174 | 0.9904225 |

两个阶段的门控输出层参数均已从零更新，排除了这部分参数完全未学习的情况；但在这些样本中，权重仍接近 1 且变化很小，表现为近似固定缩放。抽样中投影均有效，硬投影屏蔽未触发。以上仅是这 32 帧的观测，不能推广为所有场景，也没有测试坏图像下的响应。

近恒定门控与接近对照的结果相符，但不是全部精度差异的因果证明。下一版应先验证门控为何缺少区分能力，例如初始近饱和权重与训练信号的影响，再考虑受控图像退化、保持清晰样本行为的约束及相应对照；不依据本次微小差异直接追加长时间训练或叠加模块。

第一版源码与结果保留；本次没有修改算法、训练配置或启动下一轮训练。后续版本应独立记录，不能覆盖这一轮结果。

实验及 checkpoint 哈希：[门控实验记录](repro/reliability_fusion/gate-20261030_152139.json)；抽样索引和完整分布：[门控诊断](repro/reliability_fusion/gate-probe-20261030_162211.json)。完整 epoch 历史仍在运行目录 `reliability_result.json` 中。

## 下一阶段：成对图像退化评测

诊断分支：`analysis/reliability-robustness`，从已固定的 `experiment/reliability-gate-v1` 创建。本阶段仅新增诊断工具，不修改模型、checkpoint、训练配置或原始数据。

新增文件：

- [tools/analysis_tools/reliability_robustness.py:1](tools/analysis_tools/reliability_robustness.py#L1)：成对精度与门控分布诊断、固定图像退化、结果归档和本 README 更新。
- [tests/test_reliability_robustness.py:1](tests/test_reliability_robustness.py#L1)：5 项检查，验证图像变化、尺寸/类型、确定性及雷达/标注/几何不变。

条件在评测前固定：`clean` 原图；`dark` 像素乘 0.2；`blur` 21×21 Gaussian、sigma=5；`occlusion` 中心 50% 宽×50% 高区域置零（约 25% 面积）。可选 `blackout` 将全部像素置零，但仍执行图像分支，并非雷达单模态回退。上述变化只在图像解码后的内存中执行，不改变几何尺寸、雷达点或标注，也不写回数据集。

两组分别严格加载各自第 6 epoch checkpoint，采用相同均匀样本索引、条件和随机种子。先使用 32 帧排查门控是否响应图像变化；子集 AP 仅用于诊断。完整确认时使用 `--samples 1296`。合成退化结果不能直接称作真实夜间或恶劣天气性能。

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
git switch analysis/reliability-robustness
python tests/test_reliability_robustness.py

RUN="work_dirs/reliability_fusion/robustness/$(date +%Y%m%d_%H%M%S)_full"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/analysis_tools/reliability_robustness.py \
  --samples 1296 --conditions clean dark blur occlusion \
  --work-dir "$RUN" > "$RUN/console.log" 2>&1 < /dev/null &
echo $! > "$RUN/process.pid"
tail -f "$RUN/console.log"
```

脚本顺序评测两组的全部条件，全部完成后生成 `summary.json` 并更新下方独立诊断区；历史摘要保存到 `repro/reliability_fusion/robustness/`，训练结果表不受覆盖。`progress.json` 是未完成运行的进度，不能当成完整结果。完整日志、解析配置和每种条件的预测保留在运行目录。

判断时同时检查退化后 AP、相对各自 clean 的下降，以及有效投影体素的门控均值/标准差。门控低不等于预测更准确，是否有效仍需精度支持。门控统计会重新计算该小模块以读取实际分数，因此本脚本不能用作 FPS 基准。
