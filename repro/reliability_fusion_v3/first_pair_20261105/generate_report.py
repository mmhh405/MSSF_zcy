"""Archive the first V3 paired experiment and fixed-weight diagnostics."""
import argparse
import hashlib
import json
import pathlib
import shutil
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch')
parser.add_argument('--resources')
parser.add_argument('--archive', default='repro/reliability_fusion_v3/first_pair_20261105')
args = parser.parse_args()
root = pathlib.Path.cwd()
base = pathlib.Path(args.batch or (root / 'work_dirs/reliability_fusion_v3/last_paired_path.txt').read_text().strip())
res = pathlib.Path(args.resources or (root / 'work_dirs/reliability_fusion_v3/last_resource_path.txt').read_text().strip())
archive = root / args.archive
archive.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def accuracy(folder):
    record = read(folder / 'reliability_result.json')
    values = record['history'][-1]['metrics']
    def get(suffix):
        return next(v for k, v in values.items() if k.endswith(suffix))
    result = dict(eaa=get('entire_area/map'), dc=get('roi/map'))
    for cls in ('Car', 'Pedestrian', 'Cyclist'):
        result[cls] = get('entire_area/' + cls + '_3d_all')
        result[cls + '_r40'] = get('entire_area/' + cls + '_3d_all_r40')
    result['eaa_r40_mean'] = statistics.mean(result[cls + '_r40'] for cls in ('Car', 'Pedestrian', 'Cyclist'))
    return result, record


models = {}
for name, folder in [('control', base / 'dense_control'), ('v3', base / 'gate_main'),
    ('radar100', res / 'diagnostic_radar100'), ('global0625', res / 'diagnostic_global0625'),
    ('quality_first', res / 'diagnostic_quality_first'),
    ('image_budget_only', res / 'diagnostic_image_budget_only')]:
    models[name], record = accuracy(folder)
    (archive / (name + '_evaluation.json')).write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

rounds = [read(res / ('round' + str(i) + '.json')) for i in range(1, 5)]
controls = [read(res / ('trained_control' + str(i) + '.json')) for i in range(1, 3)]
quality = [read(res / ('quality_resource' + str(i) + '.json')) for i in range(1, 3)]
image_only = [read(res / ('image_only_resource' + str(i) + '.json')) for i in range(1, 3)]
for record in rounds + controls + quality + image_only:
    assert record['performance_comparison_valid'] and not record['uninitialized_router']
    assert not record['missing_keys'] and not record['competing_compute_clients_at_end']
    assert record['warmup'] == 10
    for mode in record['modes'].values():
        assert mode['samples'] == 128
        assert [row['frame_index'] for row in mode['rows']] == list(range(10, 138))
        assert all(row['predictions_finite'] for row in mode['rows'])
assert [row['mode_order'] for row in rounds] == [['dense', 'select'], ['select', 'dense']] * 2
assert len({r['checkpoint_sha256'] for r in rounds + quality + image_only}) == 1
assert len({r['checkpoint_sha256'] for r in controls}) == 1


def cost(records, mode):
    medians = [r['modes'][mode]['median_wall_ms'] for r in records]
    return dict(repeats=len(records), median_ms=statistics.median(medians),
                min_run_median_ms=min(medians), max_run_median_ms=max(medians),
                peak_allocated_mib=max(r['modes'][mode]['max_peak_allocated_bytes'] for r in records) / 2**20)


costs = dict(control=cost(controls, 'dense'), v3=cost(rounds, 'select'),
             same_weight_dense=cost(rounds, 'dense'), quality_first=cost(quality, 'select'),
             image_budget_only=cost(image_only, 'select'))
for name in ('v3', 'quality_first', 'image_budget_only'):
    costs[name]['latency_reduction_pct'] = 100 * (1 - costs[name]['median_ms'] / costs['control']['median_ms'])
    costs[name]['peak_allocated_reduction_pct'] = 100 * (1 - costs[name]['peak_allocated_mib'] / costs['control']['peak_allocated_mib'])
    records = dict(v3=rounds, quality_first=quality, image_budget_only=image_only)[name]
    rows = [row for record in records for row in record['modes']['select']['rows']]
    costs[name]['mean_image_pixel_work_ratio'] = statistics.mean(row['image_pixel_work_ratio'] for row in rows)
    costs[name]['mean_fpn_storage_ratio'] = statistics.mean(row['fpn_storage_ratio'] for row in rows)

report = dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    batch_directory=str(base), resource_directory=str(res), models=models, costs=costs,
    model_checkpoint_sha256=dict(control=controls[0]['checkpoint_sha256'], v3=rounds[0]['checkpoint_sha256']),
    protocol=dict(training_seed=0, fixed_epoch=6, full_validation_samples=1296,
        timing_samples_per_mode_per_repeat=128, timing_warmup=10, batch_size=1,
        cpu_threads=4, gpu=rounds[0]['gpu_name'], torch=rounds[0]['torch_version'],
        cuda=rounds[0]['cuda_version'], precision='float32',
        latency_scope=rounds[0]['latency_scope'], memory_scope='PyTorch peak allocated; not all driver allocations',
        diagnostic_scope='Inference-only changes on the same trained V3 checkpoint; not retrained models'),
    caveats=['One training seed only', 'Timing repeats are not independent training seeds',
        'Budgets are explored on the development validation set',
        'Quality-first results below use existing V3 weights, not a new six-epoch trained model',
        'Checkpoint size and original dataset disk usage are not reduced by activation selection',
        'Current uniform reserve samples indices and does not guarantee physical spatial stratification'])
(archive / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
for path in sorted(res.glob('*.json')):
    shutil.copyfile(path, archive / path.name)
source_files = ['tools/analysis_tools/profile_reliability_v3.py',
    'configs/experiments/reliability_fusion_v3/gate_quality_first.py',
    'configs/experiments/reliability_fusion_v3/gate_image_budget_only.py',
    'mmdet3d/models/detectors/voxelnet_fuse_v3.py',
    'mmdet3d/models/layers/fusion_layers/early_budget_gate_v3.py']
(archive / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256((root / p).read_bytes()).hexdigest()
    for p in source_files}, indent=2) + '\n', encoding='utf-8')

lines = ['# V3 首轮结果与下一步', '',
    '记录日期为 2026-11-05。服务器为 192.168.1.144，项目为 `/home/d514/zcy/MSSF_zcy`。', '',
    '## 实验内容与方法', '',
    '比较同一 V2 权重初始化、相同增强、seed=0、六轮微调的完整计算对照与 V3。采用固定第 6 轮权重和 1,296 帧验证集。主要指标为普通 EAA/DC 3D mAP，另列三类 R40 AP 的均值作为补充。', '',
    '资源在空闲 RTX 4090 上以 float32、batch size=1、4 个 CPU 线程测量。每次先预热 10 帧，再计时相同的 128 帧。V3 同权重完整/选择执行做四次交替顺序复测；实际训练的对照权重另测两次。延迟包含预处理、路由、特征提取与解码，不含数据加载。显存为 PyTorch peak allocated，不代表驱动记录的全部显存。', '',
    '## 实验结果', '',
    '| 模型 | EAA mAP | DC mAP | 汽车 AP | 行人 AP | 骑行者 AP | EAA 三类 R40 均值 |',
    '|---|---:|---:|---:|---:|---:|---:|']
names = dict(control='匹配对照，第6轮', v3='原 V3，第6轮', radar100='固定权重，雷达100%',
             global0625='固定权重，全局62.5%', quality_first='固定权重，雷达95%＋全局62.5%',
             image_budget_only='固定权重，雷达100%＋全局62.5%')
for name, value in models.items():
    lines.append('| ' + names[name] + ' | ' + ' | '.join(f'{value[key]:.4f}' for key in
        ['eaa', 'dc', 'Car', 'Pedestrian', 'Cyclist', 'eaa_r40_mean']) + ' |')
lines += ['', '后面四行是已有 V3 权重的推理诊断，不是重新训练后的消融结论。所有诊断仍在同一开发验证集上选取参数。', '',
    '| 执行配置 | 延迟中位数 ms | 各次中位数范围 ms | 峰值 allocated MiB | 延迟减少 | allocated 减少 |',
    '|---|---:|---:|---:|---:|---:|']
for name in ('control', 'v3', 'quality_first', 'image_budget_only'):
    c = costs[name]
    lines.append(f"| {names[name]} | {c['median_ms']:.2f} | {c['min_run_median_ms']:.2f}–{c['max_run_median_ms']:.2f} | {c['peak_allocated_mib']:.2f} | "
        + ('— | — |' if name == 'control' else f"{c['latency_reduction_pct']:.2f}% | {c['peak_allocated_reduction_pct']:.2f}% |"))
v = models['v3']
c = models['control']
q = models['quality_first']
image = models['image_budget_only']
lines += ['', f"原 V3 的 EAA/DC 相对对照变化为 {v['eaa']-c['eaa']:+.4f}/{v['dc']-c['dc']:+.4f} 个百分点。它已有资源收益，但尚未实现两个主要精度指标同时提高。",
    '', '## 诊断与下一步', '',
    '雷达保留率提高到 100% 后，行人普通 AP 明显回升，但 DC 没有同步恢复。全局图像比例提高到 62.5% 后，汽车 AP 回升。它们提示雷达筛选和图像尺度都影响精度，但固定权重诊断不能替代匹配训练。', '',
    f"联合候选在已有权重上取得 EAA/DC {q['eaa']:.4f}/{q['dc']:.4f}，相对对照为 {q['eaa']-c['eaa']:+.4f}/{q['dc']-c['dc']:+.4f} 个百分点。",
    '', f"保留全部雷达且使用 62.5% 全局图像的诊断取得 EAA/DC {image['eaa']:.4f}/{image['dc']:.4f}，相对对照为 {image['eaa']-c['eaa']:+.4f}/{image['dc']-c['dc']:+.4f} 个百分点。",
    '', '下一步先单独微调 `gate_image_budget_only.py`，建立保留雷达几何的图像省算对照。从原 V2 热启动权重重新训练六轮、seed=0，复用本轮匹配对照作第一轮参考。它保持图像选区 20%，保留全部雷达 pillar，全局图像比例为 62.5%。这一步用于定位物理删点的代价，不能据此宣称完整双向硬裁剪已验证成功。新候选的正式训练尚未启动。', '',
    '```bash', 'source /home/d514/zcy/MSSF_zcy/activate.sh',
    'RUN="$PWD/work_dirs/reliability_fusion_v3/image_budget_only_$(date +%Y%m%d_%H%M%S)"',
    'mkdir -p "$RUN"',
    'printf "%s\\n" "$RUN" > work_dirs/reliability_fusion_v3/last_image_budget_path.txt',
    'nohup "$MSSF_PYTHON" -u tools/train.py \\',
    '  configs/experiments/reliability_fusion_v3/gate_image_budget_only.py \\',
    '  --work-dir "$RUN" --cfg-options randomness.seed=0 \\',
    '  > "$RUN/console.log" 2>&1 < /dev/null &',
    'printf "%s\\n" "$!" > "$RUN/train.pid"', 'echo "$RUN"', '```', '',
    '训练完成后对固定第 6 轮权重检查普通 AP、R40、DC、逐类召回及资源，再与 95% 雷达保留候选做匹配训练比较。若物理删点持续损害精度，后续 V3.1 应保留低成本雷达几何通路，让门控限制昂贵的图像采样/跨模态更新。达到可接受的精度—资源折中后，再开展 seed=1、2 和独立/单向先验的匹配预算消融。不要用当前推理诊断的分数代替新候选的训练结果。', '',
    '## 限制与归档', '',
    '目前只有一个训练 seed。四次计时复测不能解释为四次训练复验。固定第六轮规则保持一致，没有单独挑选 V3 的最佳 epoch。', '',
    '当前保留策略的均匀抽样沿索引执行，不能保证真实坐标空间的均匀覆盖。未来如要宣称空间分层保护，应增加基于坐标的保留策略及对应验证。', '',
    '本轮训练经历服务器重启，已完成对照得到保留，V3 从同一 V2 权重重新跑满六轮，损坏的空检查点没有用于续训。资源在 11 月 5 日统一环境下重新测量。', '',
    f'原始运行目录为 `{base}`。资源与诊断目录为 `{res}`。原始指标、计时逐帧数据、权重哈希和源码哈希归档在 `repro/reliability_fusion_v3/first_pair_20261105/`。', '']
markdown = '\n'.join(lines)
(archive / 'README.md').write_text(markdown, encoding='utf-8')
readme = root / 'README_reliability_fusion_v3.md'
text = readme.read_text(encoding='utf-8')
text = text.replace('本节只记录本版本真实执行的验证。检查详情见 `repro/reliability_fusion_v3/verification.json`。正式 V3 精度和独占 GPU 性能结果尚待训练后取得。',
    '本节保留接入检查记录。首轮正式训练及空闲 GPU 资源测量已于 2026-11-05 汇总，见下方首轮结果链接。')
text = text.replace('空间均匀采样名额', '按索引分散抽样名额（尚非严格空间分层）')
section = '\n## 7. 首轮正式结果与下一步（2026-11-05）\n'
text = text.split(section)[0]
text += section + '\n'
text += f"固定第六轮的 V3 EAA/DC 为 {v['eaa']:.4f}/{v['dc']:.4f}，对照为 {c['eaa']:.4f}/{c['dc']:.4f}。"
text += f"资源复测中，V3 延迟减少 {costs['v3']['latency_reduction_pct']:.2f}%，峰值 allocated 显存减少 {costs['v3']['peak_allocated_reduction_pct']:.2f}%。"
text += '当前收益是精度与资源的折中。完整结果、R40 补充、固定权重诊断和下一轮可执行命令见 [首轮实验报告](repro/reliability_fusion_v3/first_pair_20261105/README.md)。\n'
readme.write_text(text, encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
