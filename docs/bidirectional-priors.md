# MSSF 双向先验实验

基线为官方 MSSF 的 `0c9482f8b0d9dc1c43b17a68fd5312bad97e7ea7`。实验配置继承仓库内的 VoD VoxelNeXt 单阶段消融 `vx_mm_r50_ablation_n1m5.py`，在 `x_conv1` 的多尺度可变形融合中启用两条辅助路径。它是用于测量改动影响的同配置基线，论文主表的 MSSF-V 数值需用作者发布的对应配置和权重复现。

## 相关工作

| 工作 | 主要做法 |
| --- | --- |
| MSSF | 在雷达稀疏骨干中投影体素质心，再从多尺度图像特征采样。语义引导头对体素做前景预测和重加权。[论文](https://arxiv.org/abs/2411.15016) |
| SGDet3D | Geometric Depth Completion 用雷达几何辅助图像深度补全和 BEV 提升。Semantic Radar PillarNet 用图像语义丰富雷达柱特征。随后用定位感知交叉注意力让 BEV 查询提取图像信息。[论文](https://doi.org/10.1109/LRA.2024.3513041) · [作者代码](https://github.com/shawnnnkb/SGDet3D) |
| BEV-radar | 分别构建雷达与图像 BEV 特征，并在解码器中用双向空间注意力交换信息。[论文](https://just.ustc.edu.cn/article/doi/10.52396/JUSTC-2023-0006) |
| RCACF | 用雷达辅助图像特征提取和动态对齐，再用热图先验引导双流可变形交叉注意力。[论文](https://www.sciencedirect.com/science/article/pii/S0031320326007430) |
| SGDet3D++ | 把雷达回波关联到 3D 锚框，以其几何支撑约束可变形图像采样并细化锚框。用多普勒核查时序匹配。[预印本](https://arxiv.org/abs/2609.27671) |
| MLF-4DRCNet | 用图像实例提示加密雷达点，再进行全局融合和候选框局部细化。论文把 MSSF 列为对比方法。[论文](https://arxiv.org/html/2509.18613v1) |

截至 2026-11-03，公开检索找到了双向融合与几何约束采样的先例，尚未找到在 MSSF 的稀疏采样块中用相机深度缩放采样偏移、再用单标量图像语义门控回写的同一实现。

## 实现

- **雷达 → 图像**。保留原有质心投影时已经算出的相机深度 `z`。可变形采样仍围绕原参考点进行，偏移乘以 `clamp(20 m / max(z, 1 m), 0.5, 2.0)`。近处搜索窗口更宽，远处更窄。无有效深度时取 1，包括稀疏卷积产生但没有原始雷达点支撑的补充体素。20 m 和截断范围是这次实验的固定初值。
- **图像 → 雷达**。从现有采样输出 `u` 计算标量门 `1 + tanh(Wu + b)`，只缩放图像更新量，再按原融合方式保留查询残差。线性层零初始化，初始倍率为 1。门控依靠原检测训练信号，使用的是图像特征所携带的隐式语义。
- 现有骨干、图像金字塔和体素检测头继续提供特征与监督。每个启用阶段的语义门增加 `C + 1` 个参数，当前配置 `C = 32`。几何路径无可学习参数。新增状态以稀疏体素标量为主，普通前向省去整批采样坐标和权重的可视化副本。可变形注意力内部仍需采样坐标和权重。

## 训练与评估

按根目录 README 准备 Linux/CUDA 环境、VoD 数据和索引，以及配置指定的 ResNet/FPN 预训练权重。根目录的环境文件名为 `environment.yaml`。在仓库根目录运行下面的命令，示例使用 Bash。

```bash
cfg=configs/voxelnext/vx_mm_r50_bidirectional_priors_vod.py
opt=model.pts_middle_encoder.optional_cfg

python tools/train.py "$cfg" --work-dir work_dirs/vod_base \
  --cfg-options "$opt.geometry_prior=False" "$opt.semantic_prior=False" randomness.seed=42
python tools/train.py "$cfg" --work-dir work_dirs/vod_geom \
  --cfg-options "$opt.semantic_prior=False" randomness.seed=42
python tools/train.py "$cfg" --work-dir work_dirs/vod_sem \
  --cfg-options "$opt.geometry_prior=False" randomness.seed=42
python tools/train.py "$cfg" --work-dir work_dirs/vod_both \
  --cfg-options randomness.seed=42

python tools/test.py "$cfg" --checkpoint work_dirs/vod_both/epoch_40.pth --metric VODMetric
```

评估前三组时复用各组训练时的两个开关取值。保持数据划分、权重初始化来源、训练周期与硬件一致。记录 VoD 的全区域及驾驶走廊 3D mAP、各类别 AP、峰值显存和同步后的单帧推理延迟。关闭两条先验时可严格加载原模型权重。启用语义门后若从原模型继续训练，使用权重初始化而非恢复优化器状态，新增门保留零初始化。

本地验证已覆盖配置继承、两个开关、关闭先验时的注意力数值一致性、无有效深度回退、批次空项、诊断开关以及语义门反向传播。张量验证用真实 PyTorch 和 MMCV 2.0.1 的纯 PyTorch 采样函数，隔离了稀疏卷积和 CUDA 扩展。端到端训练、mAP、延迟和显存数据待在完整环境中实测。

## Git

```bash
git fetch origin
git switch experiment/bidirectional-priors
git log --oneline main..HEAD
git diff main...HEAD
git push -u origin experiment/bidirectional-priors
```

新克隆且远端已有分支时可用 `git switch --track origin/experiment/bidirectional-priors`。对照实验记录使用的提交 ID、配置覆盖和随机种子。
