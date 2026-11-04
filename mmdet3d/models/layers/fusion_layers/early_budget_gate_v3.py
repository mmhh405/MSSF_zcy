"""Cheap bidirectional routing before the expensive MSSF feature extractors.

Routing scores decide execution. V2 gates continue to decide fusion weights.
No labels or augmentation-quality flags enter ``forward``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .voxel_fusion import proj_to_img
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type


def select_with_reserves(joint, own, cross, keep_ratio, reserve_fraction=.25,
                         minimum=1):
    """Exact per-frame budget; own/cross and uniform reserves can each retain items."""
    n = joint.numel()
    if not n:
        return torch.zeros_like(joint, dtype=torch.bool)
    k = min(n, max(minimum, math.ceil(n * keep_ratio)))
    mask = torch.zeros(n, dtype=torch.bool, device=joint.device)
    if k == n:
        return ~mask
    reserve = min(k // 3, max(1, int(k * reserve_fraction)))
    if reserve:
        uniform = torch.linspace(0, n - 1, reserve, device=joint.device).round().long()
        mask[uniform] = True
    for score in (own, cross):
        count = min(reserve, k - int(mask.sum()))
        if count:
            ids = score.detach().masked_fill(mask, -torch.inf).topk(count).indices
            mask[ids] = True
    count = k - int(mask.sum())
    if count:
        mask[joint.detach().masked_fill(mask, -torch.inf).topk(count).indices] = True
    return mask


def project(points, meta):
    matrix = points.new_tensor(get_proj_mat_by_coord_type(meta, 'LIDAR'))
    uvz = proj_to_img(points, meta, matrix, 'LIDAR', with_depth=True, normalize=True)
    h, w = meta['input_shape'][:2]
    ih, iw = meta.get('img_shape', (h, w))[:2]
    valid = (torch.isfinite(uvz).all(-1) & (uvz[:, 2] > 1e-5)
             & (uvz[:, 0] >= 0) & (uvz[:, 0] < iw / w)
             & (uvz[:, 1] >= 0) & (uvz[:, 1] < ih / h))
    return torch.nan_to_num(uvz), valid


def points_in_boxes(points, boxes):
    """Training targets using LiDAR bottom-centred boxes, without CUDA-only ops."""
    if not len(boxes) or not len(points):
        return points.new_zeros(len(points))
    center = boxes.gravity_center.to(points)
    dims = boxes.dims.to(points)
    yaw = boxes.yaw.to(points)
    delta = points[:, None, :3] - center[None]
    c, s = yaw.cos()[None], yaw.sin()[None]
    local = torch.stack((delta[..., 0] * c + delta[..., 1] * s,
                         -delta[..., 0] * s + delta[..., 1] * c,
                         delta[..., 2]), -1)
    return (local.abs() <= dims[None] / 2 + .15).all(-1).any(-1).float()


class BidirectionalBudgetRouter(nn.Module):
    def __init__(self, preview_size=(160, 256), hidden_channels=16,
                 tile_size=256, image_keep_ratio=.2, radar_keep_ratio=.8,
                 reserve_fraction=.25, min_pillars=32,
                 voxel_size=(.16, .16, 5.),
                 point_cloud_range=(0., -25.6, -3., 51.2, 25.6, 2.),
                 rcs_index=3, velocity_index=5,
                 use_radar_to_image=True, use_image_to_radar=True,
                 supervision_weight=.2, budget_weight=.01):
        super().__init__()
        if not (0 < image_keep_ratio <= 1 and 0 < radar_keep_ratio <= 1):
            raise ValueError('Keep ratios must be in (0, 1]')
        if not 0 <= reserve_fraction <= 1 / 3:
            raise ValueError('Each of the three reserves must fit the budget')
        self.preview_size = tuple(preview_size)
        self.tile_size = int(tile_size)
        self.image_keep_ratio = image_keep_ratio
        self.radar_keep_ratio = radar_keep_ratio
        self.reserve_fraction = reserve_fraction
        self.min_pillars = int(min_pillars)
        self.voxel_size, self.point_cloud_range = voxel_size, point_cloud_range
        self.rcs_index, self.velocity_index = rcs_index, velocity_index
        self.use_radar_to_image = use_radar_to_image
        self.use_image_to_radar = use_image_to_radar
        self.supervision_weight, self.budget_weight = supervision_weight, budget_weight
        self.preview = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GroupNorm(2, 8), nn.SiLU(),
            nn.Conv2d(8, hidden_channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, hidden_channels), nn.SiLU())
        self.visual_head = nn.Conv2d(hidden_channels, 1, 1)
        self.image_head = nn.Conv2d(hidden_channels + 3, 1, 1)
        self.radar_head = nn.Sequential(
            nn.Linear(hidden_channels + 9, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, 1))
        # Start with explicit physical/visual priors, then learn supervised utility.
        for layer in (self.visual_head, self.image_head, self.radar_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _descriptors(self, means, counts):
        rcs = means[:, self.rcs_index] if means.shape[1] > self.rcs_index else counts * 0
        speed = means[:, self.velocity_index] if means.shape[1] > self.velocity_index else counts * 0
        radius = means[:, :2].norm(dim=-1)
        desc = torch.stack((means[:, 0] / 51.2, means[:, 1] / 25.6,
                            means[:, 2] / 5., torch.log1p(counts) / math.log(33),
                            (rcs / 20).tanh(), (speed / 10).tanh(), radius / 75), -1)
        own = .4 * torch.log1p(counts) / math.log(33) + .3 * (speed.abs() / 10).tanh()
        own = own + .3 * torch.exp(-radius / 30)
        return desc, own, speed.abs()

    def forward(self, imgs, voxel_dict, metas):
        if imgs.ndim != 4 or imgs.shape[0] != len(metas):
            raise ValueError('V3 currently supports one camera per sample, BCHW')
        h, w = imgs.shape[-2:]
        for meta in metas:
            meta['input_shape'] = (h, w)
        preview = F.interpolate(imgs, size=self.preview_size, mode='bilinear', align_corners=False)
        feat = self.preview(preview)
        visual_logits = self.visual_head(feat)
        # Local contrast is a cheap quality cue, not a calibrated reliability probability.
        grey = preview.mean(1, keepdim=True)
        variance = (F.avg_pool2d(grey.square(), 5, 1, 2)
                    - F.avg_pool2d(grey, 5, 1, 2).square()).clamp_min(0)
        quality = F.adaptive_avg_pool2d(variance.sqrt(), feat.shape[-2:]).clamp(0, 1)
        voxels, counts, coors = (voxel_dict[k] for k in ('voxels', 'num_points', 'coors'))
        valid_slots = torch.arange(voxels.shape[1], device=voxels.device)[None] < counts[:, None]
        means = (torch.nan_to_num(voxels, nan=0., posinf=0., neginf=0.)
                 * valid_slots[..., None]).sum(1) / counts[:, None].clamp_min(1)
        desc, own, speed = self._descriptors(means, counts.float())
        radar_maps, point_inputs, projections = [], [], []
        fh, fw = feat.shape[-2:]
        for batch, meta in enumerate(metas):
            ids = torch.where(coors[:, 0].long() == batch)[0]
            uvz, valid = project(means[ids, :3], meta)
            # Invalid camera evidence contributes zero; radar descriptors remain available.
            grid = uvz[:, :2].clamp(-2, 2).mul(2).sub(1).view(1, 1, -1, 2)
            if len(ids):
                image_context = F.grid_sample(torch.cat((feat[batch:batch+1],
                    visual_logits[batch:batch+1].sigmoid(), quality[batch:batch+1]), 1),
                    grid, align_corners=False)[0, :, 0].T
            else:
                image_context = feat.new_empty((0, feat.shape[1] + 2))
            image_context = image_context * valid[:, None]
            if not self.use_image_to_radar:
                image_context = image_context * 0
            else:
                image_context = image_context * image_context[:, -1:].detach()
            point_inputs.append((ids, torch.cat((desc[ids], image_context), -1)))
            cell = (uvz[:, 1].clamp(0, 1-1e-6).mul(fh).long() * fw
                    + uvz[:, 0].clamp(0, 1-1e-6).mul(fw).long())
            density = feat.new_zeros(fh * fw)
            velocity = torch.zeros_like(density)
            proximity = torch.zeros_like(density)
            density.index_add_(0, cell[valid], torch.ones_like(speed[ids][valid]))
            velocity.index_add_(0, cell[valid], (speed[ids][valid] / 10).clamp_max(1))
            proximity.index_add_(0, cell[valid], torch.exp(-means[ids][valid, :2].norm(dim=-1) / 30))
            radar_map = torch.stack((density.log1p().clamp_max(3) / 3,
                velocity / density.clamp_min(1), proximity / density.clamp_min(1))).view(3, fh, fw)
            # Expand uncertain projections locally; a missing radar hit is not a veto.
            radar_maps.append(F.max_pool2d(radar_map[None], 3, 1, 1)[0])
            projections.append((ids, uvz, valid))
        radar_maps = torch.stack(radar_maps)
        image_prior = radar_maps if self.use_radar_to_image else radar_maps * 0
        image_logits = self.image_head(torch.cat((feat, image_prior), 1))
        gh, gw = math.ceil(h / self.tile_size), math.ceil(w / self.tile_size)
        image_joint = F.adaptive_max_pool2d(image_logits.sigmoid(), (gh, gw))
        image_own = F.adaptive_max_pool2d(visual_logits.sigmoid() + .1 * quality, (gh, gw))
        image_cross = F.adaptive_max_pool2d(image_prior.sum(1, keepdim=True), (gh, gw))
        image_mask = []
        radar_mask = torch.zeros(len(coors), dtype=torch.bool, device=coors.device)
        radar_logits = feat.new_zeros(len(coors))
        for batch, (ids, context) in enumerate(point_inputs):
            logits = self.radar_head(context).squeeze(-1)
            radar_logits = radar_logits.index_copy(0, ids, logits)
            cross = context[:, -2]  # quality-weighted visual objectness
            joint = logits.sigmoid() + .25 * own[ids] + .25 * cross
            keep = select_with_reserves(joint, own[ids], cross, self.radar_keep_ratio,
                                       self.reserve_fraction, self.min_pillars)
            radar_mask[ids] = keep
            image_mask.append(select_with_reserves(
                (image_joint[batch] + .5 * image_cross[batch]).flatten(),
                image_own[batch].flatten(), image_cross[batch].flatten(),
                self.image_keep_ratio, self.reserve_fraction).view(gh, gw))
        return dict(image_mask=torch.stack(image_mask), radar_mask=radar_mask,
                    visual_logits=visual_logits, image_logits=image_logits,
                    radar_logits=radar_logits, means=means, coors=coors,
                    projections=projections, input_shape=(h, w))

    def filter_inputs(self, inputs, state):
        """Filter raw points and precomputed pillars together, retaining batch IDs."""
        keep, coors = state['radar_mask'], state['coors']
        vox = inputs['voxels']
        result = dict(inputs)
        result['voxels'] = {key: value[keep] if torch.is_tensor(value)
                            and value.ndim and value.shape[0] == len(keep) else value
                            for key, value in vox.items()}
        filtered = []
        width = round((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        for batch, points in enumerate(inputs['points']):
            xy = ((points[:, :2] - points.new_tensor(self.point_cloud_range[:2]))
                  / points.new_tensor(self.voxel_size[:2])).floor().long()
            chosen = coors[keep & (coors[:, 0].long() == batch)]
            keys = chosen[:, 2].long() * width + chosen[:, 3].long()
            valid = ((xy[:, 0] >= 0) & (xy[:, 0] < width)
                     & (xy[:, 1] >= 0) & (xy[:, 1] < round(
                         (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])))
            mask = valid & torch.isin(xy[:, 1] * width + xy[:, 0], keys)
            filtered.append(points[mask])
        result['points'] = filtered
        return result

    def auxiliary_losses(self, state, samples, metas):
        """Box supervision only during loss computation, never during routing."""
        visual_targets = torch.zeros_like(state['visual_logits'])
        radar_targets = torch.zeros_like(state['radar_logits'])
        fh, fw = visual_targets.shape[-2:]
        ys = torch.arange(fh, device=visual_targets.device) / fh
        xs = torch.arange(fw, device=visual_targets.device) / fw
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        for batch, (sample, meta) in enumerate(zip(samples, metas)):
            # Det3DDataSample.metainfo returns a copy, not the dict used by routing.
            meta = dict(meta, input_shape=state['input_shape'])
            boxes = sample.gt_instances_3d.bboxes_3d
            labels = sample.gt_instances_3d.labels_3d
            boxes = boxes[labels >= 0]
            ids = torch.where(state['coors'][:, 0].long() == batch)[0]
            with torch.no_grad():
                radar_targets[ids] = points_in_boxes(state['means'][ids, :3], boxes)
                if len(boxes):
                    uvz, _ = project(boxes.corners.reshape(-1, 3), meta)
                    uvz = uvz.view(-1, 8, 3)
                    front = (uvz[..., 2] > 1e-5).all(-1)
                    uv = uvz[front, :, :2]
                    if len(uv):
                        lo, hi = uv.amin(1).clamp(0, 1), uv.amax(1).clamp(0, 1)
                        # Cell intersection retains small projected boxes.
                        covered = ((xx[None] < hi[:, 0, None, None])
                            & (xx[None] + 1 / fw > lo[:, 0, None, None])
                            & (yy[None] < hi[:, 1, None, None])
                            & (yy[None] + 1 / fh > lo[:, 1, None, None]))
                        visual_targets[batch, 0] = covered.any(0)
        positive = visual_targets.new_tensor(4.)
        visual_loss = sum(F.binary_cross_entropy_with_logits(state[key], visual_targets,
                         pos_weight=positive) for key in ('visual_logits', 'image_logits')) / 2
        radar_loss = (F.binary_cross_entropy_with_logits(state['radar_logits'], radar_targets,
                      pos_weight=positive) if radar_targets.numel() else visual_loss * 0)
        budget = (state['image_logits'].sigmoid().mean() - self.image_keep_ratio).square()
        if radar_targets.numel():
            budget = budget + (state['radar_logits'].sigmoid().mean() - self.radar_keep_ratio).square()
        return dict(loss_v3_image_utility=visual_loss * self.supervision_weight,
                    loss_v3_radar_utility=radar_loss * self.supervision_weight,
                    loss_v3_budget=budget * self.budget_weight)


class CoarseTileImageExecutor:
    """Frozen full-view coarse FPN plus selected high-resolution tile refinements.

    Persist only coarse-resolution FPN canvases for the existing deformable
    sampler. This is approximate multi-resolution inference, requiring fine-tuning.
    It does not claim dense-equivalent convolutions or a fully sparse fusion neck.
    """
    def __init__(self, global_scale=.5, halo=32, tile_batch_size=2, blend=.5):
        if not 0 < global_scale <= 1 or halo < 0 or halo % 32:
            raise ValueError('Use global scale (0,1] and halo divisible by 32')
        if tile_batch_size < 1 or not 0 <= blend <= 1:
            raise ValueError('Invalid tile batch size or blending coefficient')
        self.global_scale, self.halo = global_scale, int(halo)
        self.tile_batch_size, self.blend = int(tile_batch_size), blend

    @torch.no_grad()
    def __call__(self, imgs, image_mask, backbone, neck):
        if any(p.requires_grad for m in (backbone, neck) if m is not None for p in m.parameters()):
            raise RuntimeError('V3 tile execution requires frozen image backbone and neck')
        b, _, h, w = imgs.shape
        gh, gw = image_mask.shape[-2:]
        ch = max(32, math.ceil(h * self.global_scale / 32) * 32)
        cw = max(32, math.ceil(w * self.global_scale / 32) * 32)

        def encode(tensor):
            result = backbone(tensor)
            return tuple(neck(result) if neck is not None else result)

        canvases = list(encode(F.interpolate(imgs, (ch, cw), mode='bilinear', align_corners=False)))
        ph = math.ceil((math.ceil(h / gh) + 2 * self.halo) / 32) * 32
        pw = math.ceil((math.ceil(w / gw) + 2 * self.halo) / 32) * 32
        padded = F.pad(imgs, (self.halo, pw, self.halo, ph))
        jobs = []
        for batch, row, col in image_mask.nonzero(as_tuple=False).tolist():
            y0, y1 = round(row * h / gh), round((row + 1) * h / gh)
            x0, x1 = round(col * w / gw), round((col + 1) * w / gw)
            jobs.append((batch, y0, y1, x0, x1))
        for start in range(0, len(jobs), self.tile_batch_size):
            group = jobs[start:start + self.tile_batch_size]
            crops = torch.cat([padded[batch:batch+1, :, y0:y0+ph, x0:x0+pw]
                               for batch, y0, _, x0, _ in group])
            features = encode(crops)
            for level, canvas in enumerate(canvases):
                oh, ow = canvas.shape[-2:]
                for j, (batch, y0, y1, x0, x1) in enumerate(group):
                    top, bottom = round(y0 * oh / h), round(y1 * oh / h)
                    left, right = round(x0 * ow / w), round(x1 * ow / w)
                    if bottom <= top or right <= left:
                        continue
                    ys = (torch.arange(top, bottom, device=imgs.device) + .5) * h / oh
                    xs = (torch.arange(left, right, device=imgs.device) + .5) * w / ow
                    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                    grid = torch.stack((2 * (xx - x0 + self.halo) / pw - 1,
                                        2 * (yy - y0 + self.halo) / ph - 1), -1)[None]
                    update = F.grid_sample(features[level][j:j+1], grid.to(features[level]),
                                           mode='bilinear', align_corners=False)
                    destination = canvas[batch:batch+1, :, top:bottom, left:right]
                    destination.mul_(1 - self.blend).add_(update, alpha=self.blend)
            del crops, features
        channels = [value.shape[1] for value in canvases]
        # For MSSF's stride-4 FPN followed by stride-8/16/32/64 levels.
        reference_bytes = sum(b * c * math.ceil(h / (4 * 2**i)) * math.ceil(w / (4 * 2**i))
                              * canvases[i].element_size() for i, c in enumerate(channels))
        actual_bytes = sum(value.numel() * value.element_size() for value in canvases)
        return tuple(canvases), dict(
            image_dense_pixels=b*h*w, image_executed_pixels=b*ch*cw+len(jobs)*ph*pw,
            image_pixel_work_ratio=(b*ch*cw+len(jobs)*ph*pw)/(b*h*w),
            image_selected_tiles=len(jobs), image_total_tiles=b*gh*gw,
            fpn_persistent_bytes=actual_bytes, fpn_dense_reference_bytes=reference_bytes,
            fpn_storage_ratio=actual_bytes/reference_bytes,
            global_image_shape=[ch, cw], tile_input_shape=[ph, pw],
            feature_shapes=[list(value.shape) for value in canvases])
