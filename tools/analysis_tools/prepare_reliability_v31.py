"""Expand the matched V3.1 screening configs and record their shared inputs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from mmengine.config import Config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    groups = ['legacy', 'index_matched', 'spatial', 'spatial_independent', 'dense']
    configs = {name: Config.fromfile(str(
        root / 'configs/experiments/reliability_fusion_v31' / (name + '.py')))
        for name in groups}
    checkpoint = Path(configs['legacy'].load_from)
    annotations = [root / 'data/vod/kitti_infos_train.pkl',
                   root / 'data/vod/kitti_infos_val.pkl']
    for path in [checkpoint] + annotations:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = dict(
        protocol='six-epoch V2-warm-start screening',
        git_commit=subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path.relative_to(root)): sha256(path)
                       for path in [
                           root / 'mmdet3d/models/layers/fusion_layers/early_budget_gate_v3.py',
                           root / 'mmdet3d/models/detectors/voxelnet_fuse_v3.py']},
        initialization=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
        annotations={str(path): sha256(path) for path in annotations},
        runs=[], status='prepared; training has not been started')
    output.mkdir(parents=True)
    (output / 'configs').mkdir()
    for seed in args.seeds:
        for name, template in configs.items():
            cfg = template.copy()
            cfg.randomness = dict(seed=seed, deterministic=False)
            cfg.work_dir = str(output / 'runs' / name / ('seed' + str(seed)))
            cfg_path = output / 'configs' / (name + '_seed' + str(seed) + '.py')
            cfg.dump(str(cfg_path))
            manifest['runs'].append(dict(
                group=name, seed=seed, config=str(cfg_path),
                config_sha256=sha256(cfg_path), work_dir=cfg.work_dir,
                checkpoint_epoch=6, expected_train_iterations=30834))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(configs=len(manifest['runs']), output=str(output))))


if __name__ == '__main__':
    main()
