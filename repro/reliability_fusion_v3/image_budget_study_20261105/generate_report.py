"""Analyze the trained geometry-preserving image-budget control."""
import argparse
import hashlib
import json
import pathlib
import shutil
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run')
parser.add_argument('--base')
parser.add_argument('--resources')
parser.add_argument('--output', default='repro/reliability_fusion_v3/image_budget_study_20261105')
args = parser.parse_args()
root = pathlib.Path.cwd()
def pointer(name):
    return (root / 'work_dirs/reliability_fusion_v3' / name).read_text().strip()
run = pathlib.Path(args.run or pointer('last_image_budget_path.txt'))
base = pathlib.Path(args.base or pointer('last_paired_path.txt'))
res = pathlib.Path(args.resources or pointer('last_image_budget_analysis_path.txt'))
out = root / args.output
out.mkdir(parents=True, exist_ok=True)
if (out / 'summary.json').exists():
    raise FileExistsError(out / 'summary.json')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

metrics, evaluations, curves = {}, {}, {}
for name, folder in [('control', base/'dense_control'), ('original_v3', base/'gate_main'), ('image_budget', run)]:
    record = read(folder/'reliability_result.json')
    s = record['summary']
    assert s['training_completed'] and s['iterations'] == 30834 and s['validation_epoch'] == 6
    assert s['samples'] == 1296 and s['random_seed'] == 0
    values = record['history'][-1]['metrics']
    def get(suffix):
        return next(v for k, v in values.items() if k.endswith(suffix))
    row = dict(eaa=get('entire_area/map'), dc=get('roi/map'))
    for cls in ['Car', 'Pedestrian', 'Cyclist']:
        row[cls] = get('entire_area/'+cls+'_3d_all')
        row[cls+'_r40'] = get('entire_area/'+cls+'_3d_all_r40')
        row[cls+'_dc'] = get('roi/'+cls+'_3d_all')
    row['r40_mean'] = statistics.mean(row[c+'_r40'] for c in ['Car','Pedestrian','Cyclist'])
    metrics[name], evaluations[name] = row, record
    curves[name] = [dict(epoch=h['epoch'], eaa=next(v for k,v in h['metrics'].items() if k.endswith('entire_area/map')),
                        dc=next(v for k,v in h['metrics'].items() if k.endswith('roi/map')))
                    for h in record['history']]

profiles = {name: [read(res/f'{name}_{i}.json') for i in [1,2]] for name in ['control','image_budget']}
costs = {}
for name, records in profiles.items():
    mode = 'dense' if name == 'control' else 'select'
    for r in records:
        assert r['performance_comparison_valid'] and not r['missing_keys'] and not r['uninitialized_router']
        assert r['warmup'] == 10 and r['cpu_threads'] == 4
        assert r['modes'][mode]['samples'] == 128
        assert all(p['predictions_finite'] for p in r['modes'][mode]['rows'])
    assert len({r['checkpoint_sha256'] for r in records}) == 1
    times = [r['modes'][mode]['median_wall_ms'] for r in records]
    costs[name] = dict(repeats=2, median_ms=statistics.median(times),
        run_medians_ms=times, peak_allocated_mib=max(r['modes'][mode]['max_peak_allocated_bytes'] for r in records)/2**20)
c, old, new = metrics['control'], metrics['original_v3'], metrics['image_budget']
deltas_control = {key:new[key]-c[key] for key in new}
deltas_old = {key:new[key]-old[key] for key in new}
latency_reduction = 100*(1-costs['image_budget']['median_ms']/costs['control']['median_ms'])
memory_reduction = 100*(1-costs['image_budget']['peak_allocated_mib']/costs['control']['peak_allocated_mib'])
audit = read(res/'protocol_audit.json')
summary = dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    paths=dict(run=str(run),base=str(base),resources=str(res)),protocol_audit=audit,
    metrics=metrics,delta_vs_control=deltas_control,delta_vs_original_v3=deltas_old,
    curves=curves,costs=costs,latency_reduction_pct=latency_reduction,
    peak_allocated_reduction_pct=memory_reduction,
    checkpoint_sha256={name:rows[0]['checkpoint_sha256'] for name,rows in profiles.items()},
    limits=['Single training seed; no statistical significance claim',
        'Resource timing uses 128 frames, 10 warmup frames, two repetitions per model with reversed order',
        'Timing excludes data loading; memory is PyTorch peak allocated, not all driver VRAM',
        'Compared with original V3, both radar retention and global image resolution changed',
        'Efficiency relative to full computation does not isolate the benefit of cross-modal routing priors'])

lines = ['# V3 保留雷达对照实验分析', '',
    '记录日期为 2026-11-05。服务器为 192.168.1.144，项目为 `/home/d514/zcy/MSSF_zcy`。', '',
    '## 实验完成情况与协议', '',
    f'本轮目录为 `{run}`。训练已正常结束，完成 6 轮、30,834 次更新及最后一轮 1,296 帧验证。第 6 轮权重可读取，所有浮点状态张量为有限值。', '',
    '三组初始化路径、训练 seed=0、batch size=1、训练轮数、检测 Top-K=300 和分数门槛 0.10 已核对一致。主结论按事先固定的第 6 轮比较。', '',
    '## 精度结果', '',
    '| 模型 | EAA 3D mAP | DC 3D mAP | EAA 三类 R40 AP 均值 |',
    '|---|---:|---:|---:|']
names = dict(control='完整计算对照',original_v3='原 V3：雷达80%/全局50%',image_budget='本轮：雷达100%/全局62.5%')
for name,row in metrics.items():
    lines.append(f"| {names[name]} | {row['eaa']:.4f} | {row['dc']:.4f} | {row['r40_mean']:.4f} |")
lines += ['', f"本轮相对原 V3 的 EAA/DC 恢复 {deltas_old['eaa']:+.4f}/{deltas_old['dc']:+.4f} 个百分点；相对完整计算对照仍为 {deltas_control['eaa']:+.4f}/{deltas_control['dc']:+.4f} 个百分点。R40 均值相对对照为 {deltas_control['r40_mean']:+.4f} 个百分点，精度差距在补充口径下同样存在。", '',
    '| 类别 | 对照 EAA AP | 本轮 EAA AP | 差值 pp | 对照 DC AP | 本轮 DC AP | 差值 pp |',
    '|---|---:|---:|---:|---:|---:|---:|']
for cls in ['Car','Pedestrian','Cyclist']:
    lines.append(f"| {cls} | {c[cls]:.4f} | {new[cls]:.4f} | {deltas_control[cls]:+.4f} | {c[cls+'_dc']:.4f} | {new[cls+'_dc']:.4f} | {deltas_control[cls+'_dc']:+.4f} |")
best_control = max(curves['control'],key=lambda h:h['eaa'])
best_new = max(curves['image_budget'],key=lambda h:h['eaa'])
lines += ['', '车辆的 DC 指标已基本追平，剩余 DC 差距主要来自行人。EAA 的汽车和行人下降，骑行者小幅提高。R40 下行人相对对照下降 1.3857 个百分点，值得优先检查选区覆盖和小目标细节。指标本身不能确认漏检发生在哪个距离段，仍需要逐目标对齐。', '',
    f"日志中的本轮最高 EAA 为第 {best_new['epoch']} 轮 {best_new['eaa']:.4f}，对照自身最高为第 {best_control['epoch']} 轮 {best_control['eaa']:.4f}。不能把本轮最佳 epoch 与对照最后一轮混合比较。当前结论仍采用双方固定第 6 轮。", '',
    '## 实际资源复测', '',
    '使用各自真实训练后的第 6 轮权重，在 RTX 4090、float32、batch size=1、4 个 CPU 线程下重新测量。每种模型两次测量，每次预热 10 帧后计时相同的 128 帧，两轮交换模型执行顺序。起止均未检测到其他计算客户端。', '',
    '| 模型 | 延迟中位数 ms | 两次中位数 ms | 测量峰值 allocated MiB |',
    '|---|---:|---|---:|']
for name,value in costs.items():
    times=' / '.join(f'{v:.2f}' for v in value['run_medians_ms'])
    lines.append(f"| {names[name]} | {value['median_ms']:.2f} | {times} | {value['peak_allocated_mib']:.2f} |")
lines += ['', f'本轮延迟减少 **{latency_reduction:.2f}%**，测量峰值 allocated 显存减少 **{memory_reduction:.2f}%**。延迟包括预处理、路由、特征提取和解码，不包含数据加载。该显存数值不是驱动记录的全部显存，也不是训练显存。', '',
    '存储收益主要在中间特征及激活，模型参数与原始数据集没有因此压缩。相比原 V3，本轮提高全局分辨率并保留全部雷达，以部分资源收益换回精度，不能将恢复量单独归因于某一项修改。', '',
    '## 结论与下一步', '',
    '**阶段性结论是资源节省成立，精度差距缩小，但还没有达到精度与成本同时优于完整计算对照。** 当前只有一个训练 seed，不能宣称差距具有统计显著性，也不能把单纯降低分辨率带来的收益全部算作双向先验的贡献。', '',
    '下一步优先做相同预算的独立路由对照：保持雷达100%、全局图像62.5%、高分辨率选区20%及原 V2 融合门控，仅关闭早期雷达→图像与图像→雷达先验。配置已准备为 `configs/experiments/reliability_fusion_v3/gate_image_budget_independent.py`，正式训练尚未启动。', '',
    '若本轮优于该对照，可支持先验在预算受限条件下有效；若持平或更差，应先修改路由效用、空间覆盖保护或高分辨率特征保留方式，再扩大多 seed 实验。现有类别指标提示优先分析行人，但尚不能仅据此确认具体失效机制。', '',
    '```bash', 'source /home/d514/zcy/MSSF_zcy/activate.sh',
    'RUN="$PWD/work_dirs/reliability_fusion_v3/image_budget_independent_$(date +%Y%m%d_%H%M%S)"',
    'mkdir -p "$RUN"',
    'printf "%s\\n" "$RUN" > work_dirs/reliability_fusion_v3/last_image_budget_independent_path.txt',
    'nohup "$MSSF_PYTHON" -u tools/train.py \\',
    '  configs/experiments/reliability_fusion_v3/gate_image_budget_independent.py \\',
    '  --work-dir "$RUN" --cfg-options randomness.seed=0 \\',
    '  > "$RUN/console.log" 2>&1 < /dev/null &',
    'printf "%s\\n" "$!" > "$RUN/train.pid"', '```', '',
    '## 原始证据', '', f'本轮训练：`{run}`。资源复测：`{res}`。本目录保存训练历史、逐类指标、逐帧计时、协议审计和权重 SHA-256。', '']
markdown='\n'.join(lines)
for name,r in evaluations.items():
    (out/f'{name}_evaluation.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
for path in res.glob('*.json'):
    shutil.copyfile(path,out/path.name)
(out/'README.md').write_text(markdown,encoding='utf-8')
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
readme=root/'README_reliability_fusion_v3.md'
text=readme.read_text(encoding='utf-8')
section='\n## 8. 保留雷达对照的正式结果（2026-11-05）\n'
assert section not in text
text+=section+'\n'+f"本轮固定第6轮 EAA/DC 为 {new['eaa']:.4f}/{new['dc']:.4f}；相对原 V3 恢复 {deltas_old['eaa']:+.4f}/{deltas_old['dc']:+.4f} pp，相对完整计算对照仍低 {abs(deltas_control['eaa']):.4f}/{abs(deltas_control['dc']):.4f} pp。实际延迟减少 {latency_reduction:.2f}%，峰值 allocated 显存减少 {memory_reduction:.2f}%。完整分析、类别差异、R40 和下一轮等预算对照命令见 [本轮分析报告](repro/reliability_fusion_v3/image_budget_study_20261105/README.md)。\n"
readme.write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
