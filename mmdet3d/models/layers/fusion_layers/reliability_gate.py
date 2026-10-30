"""Per-voxel image-update gating for MSSF; no labels are used at inference."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ReliabilityGate(nn.Module):
    """Learn a scalar gate from radar/image agreement and projection geometry.

    This score is a learned feature weight, not a calibrated probability.
    Invalid projections suppress the image update, never the query residual.
    """

    def __init__(self, channels, hidden_channels=16, init_probability=0.99):
        super().__init__()
        if not 0.0 < init_probability < 1.0:
            raise ValueError('init_probability must be strictly between 0 and 1')
        self.radar_norm = nn.LayerNorm(channels)
        self.image_norm = nn.LayerNorm(channels)
        self.radar_proj = nn.Linear(channels, hidden_channels)
        self.image_proj = nn.Linear(channels, hidden_channels)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels + 4, hidden_channels),
            nn.ReLU(), nn.Linear(hidden_channels, 1))
        # Near identity for valid points; new gates still require fine-tuning.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.constant_(self.predictor[-1].bias,
                          math.log(init_probability / (1 - init_probability)))
        self.last_stats = {}

    def forward(self, radar_features, image_update, projection_valid, geometry,
                return_gate=False):
        if radar_features.shape != image_update.shape:
            raise ValueError('radar_features and image_update must have equal shapes')
        if geometry.shape != image_update.shape[:-1] + (3,):
            raise ValueError('geometry must contain normalized u, v, and log-depth')
        valid = projection_valid.bool()
        if valid.shape == image_update.shape[:-1]:
            valid = valid.unsqueeze(-1)
        if valid.shape != image_update.shape[:-1] + (1,):
            raise ValueError('projection_valid must have one entry per query')
        valid = (valid & torch.isfinite(image_update).all(-1, keepdim=True)
                 & torch.isfinite(radar_features).all(-1, keepdim=True)
                 & torch.isfinite(geometry).all(-1, keepdim=True))
        safe_image = torch.nan_to_num(image_update, nan=0., posinf=0., neginf=0.)
        safe_radar = torch.nan_to_num(radar_features, nan=0., posinf=0., neginf=0.)
        image = self.image_proj(self.image_norm(safe_image))
        radar = self.radar_proj(self.radar_norm(safe_radar))
        agreement = F.cosine_similarity(radar, image, dim=-1, eps=1e-6).unsqueeze(-1)
        safe_geometry = torch.nan_to_num(geometry, nan=0., posinf=0., neginf=0.)
        inputs = torch.cat((radar, image, agreement, safe_geometry), dim=-1)
        gate = torch.sigmoid(self.predictor(inputs)) * valid.to(image_update.dtype)
        output = safe_image * gate
        if gate.numel():
            self.last_stats = dict(mean=gate.detach().mean(),
                                   valid_fraction=valid.float().detach().mean())
        return (output, gate) if return_gate else output
