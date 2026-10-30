"""Gate sampled image features before they enter MSSF's mixed query projection."""
import torch

from mmdet3d.registry import MODELS
from .voxel_fusion_block import VoxelFusionBlock


def build_query_gate(cfg, radar_channels, image_channels):
    if cfg is None:
        return None
    cfg = dict(cfg)
    if not cfg.pop('enabled', True):
        return None
    cfg.setdefault('type', 'ReliabilityGateV2')
    cfg.update(channels=radar_channels, image_channels=image_channels)
    gate = MODELS.build(cfg)
    if not getattr(gate, 'supports_clean_anchor', False):
        raise ValueError('Query gating requires a V2 gate with separate image channels')
    return gate


def gate_query_image(gate, radar, image, centroids, coordinates, metas):
    if gate is None:
        return image
    valid = torch.zeros((image.shape[0], 1), dtype=torch.bool, device=image.device)
    geometry = image.new_zeros((image.shape[0], 3))
    clean = torch.zeros_like(valid)
    for batch, meta in enumerate(metas):
        mask = coordinates[:, 0].long() == batch
        if not mask.any():
            continue
        projected, context = VoxelFusionBlock.projection_context(centroids[mask, 1:4], meta)
        valid[mask] = projected[0]
        geometry[mask] = context[0]
        clean[mask] = bool(meta.get('v2_is_clean', True))
    return gate(radar, image, valid, geometry, clean_mask=clean)
