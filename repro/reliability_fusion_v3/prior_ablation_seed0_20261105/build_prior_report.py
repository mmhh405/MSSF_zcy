"""Archive existing seed-0 prior ablation; never launch training."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from mmengine.config import Config

ROOT = Path.cwd()
OUT = ROOT / 'repro/reliability_fusion_v3/prior_ablation_seed0_20261105'
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


runs, configs, sources = {}, {}, {}
for arm, pointer in [('guided', 'last_image_budget_path.txt'),
                     ('independent', 'last_image_budget_independent_path.txt')]:
    run = Path((ROOT / 'work_dirs/reliability_fusion_v3' / pointer).read_text().strip())
    data = json.loads((run / 'reliability_result.json').read_text())
    s = data['summary']
    assert s['training_completed'] and s['validation_epoch'] == 6 and s['iterations'] == 30834
    assert s['random_seed'] == 0 and s['samples'] == 1296 and s['train_samples'] == 5139
    save(arm + '_result.json', data)
    cfg_path, = run.glob('*.py')
    cfg = Config.fromfile(str(cfg_path))
    configs[arm] = cfg
    (OUT / (arm + '_resolved.py')).write_text(cfg.pretty_text, encoding='utf-8')
    runs[arm] = data
    checkpoint = torch.load(run / 'epoch_6.pth', map_location='cpu')
    assert checkpoint['meta']['epoch'] == 6 and checkpoint['meta']['iter'] == 30834
    assert all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values() if v.is_floating_point())
    sources[arm] = dict(work_dir=str(run), git_commit=s['git_commit'],
                        git_dirty_at_start=s['git_dirty_at_start'],
                        result_sha256=digest(run / 'reliability_result.json'),
                        checkpoint_sha256=digest(run / 'epoch_6.pth'),
                        checkpoint_bytes=(run / 'epoch_6.pth').stat().st_size,
                        checkpoint_epoch=6, checkpoint_iter=30834, checkpoint_finite=True)
models = [copy.deepcopy(c.model) for c in configs.values()]
for model in models:
    for key in ('use_radar_to_image', 'use_image_to_radar'):
        model['early_budget']['router'].pop(key, None)
checks = dict(model_equal_except_prior=models[0] == models[1])
for key in ('train_cfg', 'train_dataloader', 'val_dataloader', 'val_evaluator',
            'optim_wrapper', 'param_scheduler', 'randomness', 'load_from'):
    checks[key + '_equal'] = configs['guided'].get(key) == configs['independent'].get(key)
shared = ['mmdet3d/models', 'mmdet3d/engine', 'configs/experiments/reliability_fusion_v3']
diff = subprocess.check_output(['git', 'diff', '--name-only', sources['guided']['git_commit'],
                                sources['independent']['git_commit'], '--', *shared], text=True).splitlines()
allowed = ['configs/experiments/reliability_fusion_v3/gate_image_budget_independent.py']
checks['no_shared_model_or_training_code_change_between_commits'] = all(p in allowed for p in diff)
assert all(checks.values()), checks
save('protocol_audit.json', dict(checks=checks, changed_paths_between_commits=diff, sources=sources,
                               note='Both runs report dirty workspaces; resolved configurations and original result files are archived.'))


def metric(arm, suffix):
    row, = [h for h in runs[arm]['history'] if h['phase'] == 'validation' and h['epoch'] == 6]
    value, = [v for k, v in row['metrics'].items() if k.endswith(suffix)]
    return value


values, deltas = {}, {}
for area in ('entire_area', 'roi'):
    for kind in ('map', 'Car_3d_all', 'Pedestrian_3d_all', 'Cyclist_3d_all',
                 'Car_3d_all_r40', 'Pedestrian_3d_all_r40', 'Cyclist_3d_all_r40'):
        key = area + '/' + kind
        values[key] = {arm: metric(arm, key) for arm in runs}
        deltas[key] = values[key]['guided'] - values[key]['independent']
save('comparison.json', dict(fixed_epoch=6, seed=0, metrics=values, guided_minus_independent_pp=deltas,
                            conclusion='Mixed small single-seed effect; paired repeat seeds 1/2 required.'))
r40_delta = {area: sum(deltas[area + '/' + cls + '_3d_all_r40']
                       for cls in ('Car', 'Pedestrian', 'Cyclist')) / 3
             for area in ('entire_area', 'roi')}
table = '\n'.join('| ' + name + ' | ' + ' | '.join(f'{x:.4f}' for x in (
    values[key]['guided'], values[key]['independent'], deltas[key])) + ' |'
    for name, key in [('EAA mAP', 'entire_area/map'), ('DC mAP', 'roi/map'),
                      ('EAA Car', 'entire_area/Car_3d_all'), ('EAA Pedestrian', 'entire_area/Pedestrian_3d_all'),
                      ('EAA Cyclist', 'entire_area/Cyclist_3d_all')])
epochs = []
for arm, data in runs.items():
    for row in data['history']:
        if row['phase'] == 'validation':
            metrics = row['metrics']
            get = lambda suffix: next(v for k, v in metrics.items() if k.endswith(suffix))
            epochs.append(f"| {arm} | {row['epoch']} | {get('entire_area/map'):.4f} | {get('roi/map'):.4f} |")
report = '''# V3 同预算先验消融与多种子复验

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
''' + table + f'''

EAA三类R40均值差为 {r40_delta['entire_area']:+.4f} pp，DC三类R40均值差为 {r40_delta['roi']:+.4f} pp。普通AP与R40均呈现EAA略升、DC略降。

| 方法 | epoch | EAA mAP | DC mAP |
|---|---:|---:|---:|
''' + '\n'.join(epochs) + '''

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
printf '%s\\n' "$BATCH" > work_dirs/reliability_fusion_v3/last_prior_repeats_path.txt
nohup env CUDA_VISIBLE_DEVICES=0 "$MSSF_PYTHON" -u \\
  tools/analysis_tools/run_v3_prior_repeats.py --work-dir "$BATCH" \\
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
'''
(OUT / 'README.md').write_text(report, encoding='utf-8')
readme = ROOT / 'README_reliability_fusion_v3.md'
original = readme.read_text(encoding='utf-8')
marker = '## 9. 同预算早期先验消融（2026-11-05）'
addition = '\n\n' + marker + '''

固定第6轮seed=0：guided EAA/DC为63.0273/81.6050，independent为62.8032/81.7427，差值+0.2241/−0.1377 pp。训练协议与计算预算一致。当前收益较小且方向不一致，需补seed=1、2。完整分析及可恢复队列命令见[本轮报告](repro/reliability_fusion_v3/prior_ablation_seed0_20261105/README.md)。新队列尚未启动。
'''
if marker not in original:
    readme.write_text(original.rstrip() + addition, encoding='utf-8')
print('ARCHIVED', OUT)
