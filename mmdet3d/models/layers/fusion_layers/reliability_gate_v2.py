"""Identity-initialized image weighting with a training-only clean anchor."""
import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ReliabilityGateV2(nn.Module):
    """Use 2*sigmoid(z), with z initially zero, as a feature weight in (0, 2).

    This is not a confidence probability. Clean/degraded metadata is used only
    to select a training regularizer; it never enters the prediction network.
    """
    supports_clean_anchor = True

    def __init__(self, channels, image_channels=None, hidden_channels=16):
        super().__init__()
        image_channels = image_channels or channels
        self.channels = channels
        self.image_channels = image_channels
        self.radar_norm = nn.LayerNorm(channels)
        self.image_norm = nn.LayerNorm(image_channels)
        self.radar_proj = nn.Linear(channels, hidden_channels)
        self.image_proj = nn.Linear(image_channels, hidden_channels)
        # Preserve raw feature-energy cues lost by per-token normalization.
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels + 6, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.last_stats = {}
        self.reset_clean_anchor()

    def reset_clean_anchor(self):
        self._anchor_sums = []
        self._anchor_counts = []

    def consume_clean_anchor(self):
        zero = self.predictor[-1].weight.sum() * 0
        total = sum(self._anchor_sums, zero)
        count = sum(self._anchor_counts, zero.detach())
        self.reset_clean_anchor()
        return total, count

    def forward(self, radar_features, image_update, projection_valid, geometry,
                return_gate=False, clean_mask=None):
        shape = image_update.shape[:-1]
        if (radar_features.shape[:-1] != shape
                or radar_features.shape[-1] != self.channels
                or image_update.shape[-1] != self.image_channels):
            raise ValueError('Radar/image dimensions do not match the configured gate')
        if geometry.shape != shape + (3,):
            raise ValueError('Expected u, v, log-depth for each query')
        valid = projection_valid.bool()
        if valid.shape == shape:
            valid = valid.unsqueeze(-1)
        if valid.shape != shape + (1,):
            raise ValueError('Expected one projection-valid flag per query')
        valid = (valid & torch.isfinite(radar_features).all(-1, keepdim=True)
                 & torch.isfinite(image_update).all(-1, keepdim=True)
                 & torch.isfinite(geometry).all(-1, keepdim=True))
        radar_raw = torch.nan_to_num(radar_features, nan=0., posinf=0., neginf=0.)
        image_raw = torch.nan_to_num(image_update, nan=0., posinf=0., neginf=0.)
        radar = self.radar_proj(self.radar_norm(radar_raw))
        image = self.image_proj(self.image_norm(image_raw))
        agreement = F.cosine_similarity(radar, image, dim=-1, eps=1e-6).unsqueeze(-1)
        radar_energy = radar_raw.float().square().mean(-1, keepdim=True).log1p().to(radar)
        image_energy = image_raw.float().square().mean(-1, keepdim=True).log1p().to(image)
        context = torch.cat((radar, image, agreement,
                             torch.nan_to_num(geometry, nan=0., posinf=0., neginf=0.),
                             radar_energy, image_energy), dim=-1)
        scores = 2 * torch.sigmoid(self.predictor(context))
        scores = scores * valid.to(scores.dtype)
        if self.training and torch.is_grad_enabled() and clean_mask is not None:
            mask = torch.broadcast_to(clean_mask.bool(), scores.shape) & valid
            self._anchor_sums.append(((scores - 1).square() * mask).sum())
            self._anchor_counts.append(mask.sum().detach())
        if scores.numel():
            self.last_stats = dict(mean=scores.detach().mean(),
                                   valid_fraction=valid.float().detach().mean())
        output = image_raw * scores
        return (output, scores) if return_gate else output
