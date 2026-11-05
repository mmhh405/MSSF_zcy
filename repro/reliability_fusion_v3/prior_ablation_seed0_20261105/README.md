# V3 同预算先验消融与多种子复验

## 1. 实验内容与方法

本轮检验早期跨模态先验是否比独立路由更有效。guided 使用雷达→图像及图像→雷达先验；independent 仅关闭这两个早期先验开关。两组均保留 V2 的融合门控、100% 雷达、0.625 全局图像缩放和20%高分辨率块预算。

当前雷达保留率为100%，图像→雷达头不会实际删除雷达点，其影响主要通过辅助训练及耦合路径产生。本实验不能单独证明物理雷达裁剪有效，也不能分别归因两个方向。共享的预算节省不能全部归因于先验。

## 2. 实验过程与一致性

两组从相同 V2 seed=0 权重出发，微调6轮、batch=1，训练5139帧、验证1296帧。训练数据与增强、优化器、学习率、检测头、验证协议经解析配置核对一致；共享模型与训练代码在两次提交间没有变化。两组工作区均报告 dirty，原始记录与解析配置保存在本目录。第6轮权重均已可加载、张量有限、epoch=6、iter=30834。

主比较固定第6轮普通3D AP；R40仅作为补充，完整指标和每轮记录见JSON。不挑选两组不同最佳轮数。本次只完成seed=0。

## 3. 实验结果

单位为AP百分数，差值为百分点；正值表示guided较高。

| 指标 | guided先验 | independent独立 | 差值 |
|---|---:|---:|---:|
| EAA mAP | 63.0273 | 62.8032 | 0.2241 |
| DC mAP | 81.6050 | 81.7427 | -0.1377 |
| EAA Car | 59.9664 | 59.3504 | 0.6160 |
| EAA Pedestrian | 50.9152 | 50.8358 | 0.0794 |
| EAA Cyclist | 78.2003 | 78.2234 | -0.0231 |

EAA三类R40均值差为 +0.1866 pp，DC三类R40均值差为 -0.0690 pp。普通AP与R40均呈现EAA略升、DC略降。

| 方法 | epoch | EAA mAP | DC mAP |
|---|---:|---:|---:|
| guided | 1 | 63.7310 | 81.5526 |
| guided | 2 | 62.6365 | 81.5704 |
| guided | 3 | 64.4014 | 81.6981 |
| guided | 4 | 62.9122 | 81.6982 |
| guided | 5 | 63.1085 | 81.6930 |
| guided | 6 | 63.0273 | 81.6050 |
| independent | 1 | 63.3713 | 81.6230 |
| independent | 2 | 62.2189 | 81.7107 |
| independent | 3 | 64.1142 | 79.5405 |
| independent | 4 | 63.8606 | 79.5911 |
| independent | 5 | 62.6970 | 79.6121 |
| independent | 6 | 62.8032 | 81.7427 |

## 4. 结论

当前先验版相对独立路由的收益较小且方向不一致，不能宣称稳定提升或统计显著。EAA的增益主要来自Car，DC行人与骑行者有所下降。已有完整计算对照63.5899/82.0461，先验版仍低0.5626/0.4411 pp。

此前先验版相对完整计算对照测得延迟减少10.29%、峰值allocated显存减少46.70%；该资源比较不是本轮guided/independent的资源差，且不表示权重文件变小。详见上一份保留雷达报告。

## 5. 下一步可执行步骤

补跑seed=1、2，每个seed分别训练guided和independent，保持本轮预算及第6轮终点。共4次微调，4090上按每次40–45分钟估计约3小时，验证和负载会影响耗时。三个seed仍共享同一份V2 seed=0预训练权重，检验的是该初始化条件下的微调随机性。

在192.168.1.144的VS Code远程终端执行一次：

```bash
source /home/d514/zcy/MSSF_zcy/activate.sh
cd /home/d514/zcy/MSSF_zcy
BATCH="$PWD/work_dirs/reliability_fusion_v3/prior_repeats_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH"
printf '%s\n' "$BATCH" > work_dirs/reliability_fusion_v3/last_prior_repeats_path.txt
nohup env CUDA_VISIBLE_DEVICES=0 "$MSSF_PYTHON" -u \
  tools/analysis_tools/run_v3_prior_repeats.py --work-dir "$BATCH" \
  > "$BATCH/queue.log" 2>&1 < /dev/null &
echo "后台进程：$!；结果目录：$BATCH"
```

查看总进度：

```bash
cd /home/d514/zcy/MSSF_zcy
BATCH="$(cat work_dirs/reliability_fusion_v3/last_prior_repeats_path.txt)"
tail -n 30 "$BATCH/queue.log"
cat "$BATCH/progress.json"
```

`START 1 guided`表示当前运行，`DONE`表示单组完成，`ALL_COMPLETED`表示4组全部完成。每组详细日志位于`seed1/guided/console.log`等目录；先确认START对应组再tail。Ctrl+C仅停止tail查看。完成后读取`comparison.json`，比较三组seed配对差值、均值与样本标准差。

中断后，激活环境，用同一BATCH再次执行上面的nohup命令即可。脚本校验源代码/初始化权重、跳过验证完整的已完成组、从有效的最近整轮checkpoint恢复，并保留旧日志。损坏checkpoint移至`rejected_checkpoints/`。如第6轮checkpoint完整但最终指标记录缺失，脚本会停止要求补评测，避免误报完成。尚未保存的半轮计算会重做，恢复不保证与未中断运行逐位一致。

若三seed收益方向仍不稳定，先分析选块覆盖与小目标漏检，再调整路由；若EAA稳定获益而DC仍下降，报告该取舍。后续再做雷达→图像/图像→雷达单向消融与鲁棒性验证，不将现有证据写成全面精度改进。

## 6. 本轮文件

- `guided_result.json`、`independent_result.json`：原始summary与6轮完整指标。
- `guided_resolved.py`、`independent_resolved.py`：实际解析的配置。
- `protocol_audit.json`：一致性、checkpoint SHA256和代码版本。
- `comparison.json`：第6轮普通AP、R40及配对差值。
- `build_prior_report.py`：本报告生成脚本。
- `tools/analysis_tools/run_v3_prior_repeats.py`：新增多种子队列工具；不修改算法。

本次只准备复验脚本并归档seed=0结果，尚未启动seed=1、2训练。
