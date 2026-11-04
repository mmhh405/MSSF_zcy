# MSSF 可靠性感知融合 V3

记录日期为 2026-11-04。V3 在 V2 的融合门控之前增加低成本双向先验路由，目标是改善精度与实际推理资源开销的折中。当前实现是待微调的研究版本。V2 的精度结果不能当作 V3 的结果。

## 1. 本版本实现的功能

- **主干前预览。** 已归一化图像先缩放到 160×256，经两层轻量卷积生成视觉预览。雷达在 PillarFeatureNet、稀疏卷积和跨模态注意力之前统计每个 pillar 的位置、点数、RCS、补偿径向速度和距离。投影沿用 MSSF 的点云增强反变换、图像缩放与翻转规则。
- **雷达→图像。** 点投影产生密度、速度和距离先验，与预览特征共同计算图像区域的执行优先级。投影图做邻域扩展。雷达没有回波的图像区域仍可由视觉分数和空间保留名额入选。
- **图像→雷达。** 在 pillar 投影处采样预览语义及局部对比度线索，和雷达自身统计一起预测 pillar 执行分数。投影无效时视觉输入清零，雷达自身分数和保留名额仍然生效。局部对比度只是质量线索，不能解释为校准后的可靠性。
- **确定的执行预算。** 图像默认选择 20% 区域，雷达默认保留 80% pillar，每帧至少保留 32 个已有 pillar。预算内分别预留本模态、另一模态和空间均匀采样名额，剩余按联合分数选择。小于 32 个 pillar 的帧全部保留。预算决定选择数量，训练正则控制连续分数分布。
- **实际选择性图像计算。** 冻结的 ResNet/FPN 先处理半分辨率全图，再只处理入选的原分辨率图像块。分块核心约 256 像素，额外 halo 为 32 像素，输入补齐到 32 的倍数，每次最多处理 2 个块。高分辨率块按原图坐标融合回低分辨率 FPN，默认混合系数 0.5。
- **实际减少早期雷达工作量。** 同步筛选原始点和预处理后的 pillar。两处稀疏雷达融合、pillar 编码和注意力的输入随之减少。后续 SECOND 稠密 BEV 主干及检测头仍按原尺寸计算。
- **区分计算选择与融合权重。** V3 路由分数选择执行区域，原 V2 的四个门控继续调节已计算图像特征的融合强度。
- **训练与推理隔离。** 3D 框仅在辅助损失中监督预览目标性及 pillar 前景性。连续分数另有预算约束。前向选择不接收 GT、遮挡标签、清晰标签或完整主干特征。选择使用硬索引，路由器通过显式辅助损失学习，不宣称检测损失对离散索引直接可导。

## 2. 资源成本边界

V3 保存和传给融合模块的是低分辨率稠密 FPN，不是全分辨率稠密 FPN。选区图像计算会生成临时分块特征，并在处理后释放。实际节省必须包含全局低分辨率计算、预览编码、投影、选区、halo 重叠和数据整理开销。

本实现不保证块卷积与原全图卷积等价。全局分辨率、边界上下文和融合特征均发生改变，需要微调与精度评测。短检查不证明精度提升或端到端加速。

`image_pixel_work_ratio` 是图像主干输入像素数量代理，包含低分辨率全图及全部分块 halo，不是 GFLOPs。`fpn_storage_ratio` 是实际 FPN 输出字节数与原 stride-4/8/16/32/64 FPN 的同精度参考字节数之比，不是整网峰值显存节省比例。完整显存和延迟由实测工具记录。

模型权重没有因空间路由而压缩，新增预览网络还会增加少量参数。原始图像与数据集磁盘占用保持原样。对“存储”的首要研究目标是中间激活、FPN 缓存和传递特征字节数。

## 3. 修改文件及位置

| 文件 | 入口与作用 |
|---|---|
| `mmdet3d/models/layers/fusion_layers/early_budget_gate_v3.py` | `BidirectionalBudgetRouter` 双向先验、保留预算、点/pillar 同步筛选及训练损失；`CoarseTileImageExecutor` 选区主干执行和低分辨率特征保存 |
| `mmdet3d/models/detectors/voxelnet_fuse_v3.py` | `VoxelNetFuseV3.extract_feat/loss` 在图像和雷达昂贵特征提取前接入选择，消费辅助损失，释放训练缓存 |
| `configs/experiments/reliability_fusion_v3/gate_main.py` | 主版本，继承 V2 增强、归一化冻结及 6 轮微调协议 |
| `configs/experiments/reliability_fusion_v3/dense_control.py` | 完整 V2 执行方式的匹配微调对照 |
| `configs/experiments/reliability_fusion_v3/radar_to_image_only.py` | 关闭图像→雷达先验，保持预算一致 |
| `configs/experiments/reliability_fusion_v3/image_to_radar_only.py` | 关闭雷达→图像先验，保持预算一致 |
| `configs/experiments/reliability_fusion_v3/independent.py` | 关闭两个跨模态先验，各自选择，保持预算一致 |
| `configs/experiments/reliability_fusion_v3/smoke_*.py` | 4 帧推理和 2 次优化器更新检查 |
| `tools/analysis_tools/profile_reliability_v3.py` | 相同输入、相同权重的 dense/select 路径计时、峰值显存、FPN 字节及实际选区数量 |
| `tests/test_reliability_fusion_v3.py` | 预算、投影、批次对齐、无效视觉证据、辅助损失梯度、实际执行调用和存储检查 |
| `repro/reliability_fusion_v3/verification.json` | 本次检查证据，随验证结果生成 |
| `repro/reliability_fusion_v3/results.json` | 后续训练与验证的实际指标索引 |

V3 通过配置的 `custom_imports` 注册，V1/V2 的模型文件和配置不改写。V2 未提交记录保留在工作区。V3 只在本项目目录工作，不修改相邻的 `MSSF_bidirectional` 项目。

## 4. 运行步骤

以下命令在服务器执行。完整训练和性能对比应在其他 GPU 训练结束后进行。每次用新的运行目录。主配置从现有 V2 seed=0 第 6 轮权重热启动，路由器此时尚未训练。

```bash
source /home/d514/miniconda3/etc/profile.d/conda.sh
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
export CUDA_VISIBLE_DEVICES=0 NUMBA_CUDA_USE_NVIDIA_BINDING=1
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
```

先检查代码与两次更新。

```bash
python tests/test_reliability_fusion_v3.py
RUN="work_dirs/reliability_fusion_v3/smoke_train/$(date +%Y%m%d_%H%M%S)"
python tools/train.py configs/experiments/reliability_fusion_v3/smoke_train.py --work-dir "$RUN"
```

正式微调必须同时跑匹配对照。两个命令顺序执行，均从同一份 V2 权重出发，并使用同一 seed。主配置仍为 6 轮，不会自动使用 smoke checkpoint。

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="work_dirs/reliability_fusion_v3/paired_$STAMP"
python tools/train.py configs/experiments/reliability_fusion_v3/dense_control.py \
  --work-dir "$BASE/dense_control" --cfg-options randomness.seed=0
python tools/train.py configs/experiments/reliability_fusion_v3/gate_main.py \
  --work-dir "$BASE/gate_main" --cfg-options randomness.seed=0
```

用完成微调的 V3 权重评估 1,296 帧验证集，再记录资源。`BASE` 必须指向本次真实的训练目录。

```bash
python tools/test.py configs/experiments/reliability_fusion_v3/gate_main.py \
  --checkpoint "$BASE/gate_main/epoch_6.pth" --work-dir "$BASE/gate_main/full_val"
python tools/analysis_tools/profile_reliability_v3.py \
  --checkpoint "$BASE/gate_main/epoch_6.pth" --samples 32 --warmup 5 \
  --output "$BASE/gate_main/resources.json"
```

资源工具的 dense/select 切换使用同一权重，用于考察执行开销。精度结论使用分别微调的匹配对照和 V3，不把“将 V3 权重直接切到 dense”的结果当作训练匹配对照。工具默认检测其他 GPU 计算进程并拒绝性能对比。`--allow-busy-gpu` 只用于标明竞争条件的接入检查。

完成 seed=0 后再复验 seed=1、2，并补齐三个先验消融。预算至少比较图像 10%/20%/30% 和雷达 60%/80%/100% 的少量组合，按验证精度—资源曲线选择候选，不能只按稀疏率宣称最优。

## 5. 研究贡献与验证要求

待检验的贡献是双向低成本先验如何共同分配执行预算、保留互补目标，并让选择真实作用于主干计算和中间特征存储。已有雷达引导 ROI、浅层分支门控、图像指导雷达筛选及块级稀疏计算工作。尤其 GATS 已在视觉主干之前用 LiDAR/雷达几何选择图像 token。因此本版本不作“首次主干前几何选择”声明。

- GATS，2026-10-29 预印本：https://arxiv.org/html/2610.40935v1
- Radar Guided Dynamic Visual Attention，2022：https://arxiv.org/abs/2206.01772
- HydraFusion，2022：https://arxiv.org/abs/2201.06644
- SparseInteraction，2024：https://openreview.net/forum?id=3XaxpSG9F8
- SBNet，2018：https://openaccess.thecvf.com/content_cvpr_2018/papers/Ren_SBNet_Sparse_Blocks_CVPR_2018_paper.pdf

首要指标为普通 EAA/DC 3D AP、行人及远距离目标召回、无雷达回波目标覆盖、端到端延迟分布、峰值 allocated 显存和实际 FPN 输出字节数。资源结果必须记录硬件、精度、batch、warmup、其他 GPU 客户端及 checkpoint 哈希。稠密 BEV 和模型参数成本应单列。

## 6. 接入检查与后续结果

本节只记录本版本真实执行的验证。检查详情见 `repro/reliability_fusion_v3/verification.json`。正式 V3 精度和独占 GPU 性能结果尚待训练后取得。

<!-- V3_VERIFICATION_START -->

2026-11-04 接入验证通过：12 项 V3 单元检查、9 项 V2 回归检查、真实 VoD 前向、2 次优化器更新、检查权重的完整重新加载和再次推理。三个新路由预测头均产生非零更新，冻结图像分支的全部 334 个状态张量与 V2 初始化权重一致。新增路由参数共 1,910 个。

代表性 1,216×1,952 帧选择 8/40 个图像块，主干输入像素代理比为 0.5992，持久 FPN 输出字节比为 0.2542。这些不是整网加速比例。两帧计时在另一个训练进程运行时取得，dense/select 中位数分别为 93.11/92.27 ms，不能据此声称加速。正式精度和空闲 GPU 性能待测。

原始检查目录为 `work_dirs/reliability_fusion_v3/verification_20261104_113049`。两步权重仅用于接入检查，不用于正式实验。

<!-- V3_VERIFICATION_END -->

下表由 `ReliabilityExperimentHook` 在训练/验证后自动更新。smoke 行只表示接入验证，不表示完成正式训练。

<!-- RELIABILITY_RESULTS_START -->

| 实验 | 类型/状态 | 样本数 | 验证 epoch | EAA mAP (%) | DC mAP (%) | 代码 | 运行目录 |
|---|---|---:|---:|---:|---:|---|---|
| v3-two-update-backward | smoke_only | — | — | — | — | 4dad3902+dirty | work_dirs/reliability_fusion_v3/verification_20261104_113049/train_smoke_fixed |

<!-- RELIABILITY_RESULTS_END -->
