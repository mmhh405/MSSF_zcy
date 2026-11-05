"""MSSF V3: bidirectional early budget routing followed by V2 fusion gates."""
import torch

from mmdet3d.registry import MODELS
from .voxelnet_fuse import VoxelNetFuse
from ..layers.fusion_layers.early_budget_gate_v3 import (
    BidirectionalBudgetRouter, CoarseTileImageExecutor)


@MODELS.register_module()
class VoxelNetFuseV3(VoxelNetFuse):
    def __init__(self, early_budget=None, **kwargs):
        super().__init__(**kwargs)
        options = dict(early_budget or {})
        self.routing_mode = options.pop('mode', 'select')
        if self.routing_mode not in ('select', 'dense'):
            raise ValueError('V3 routing mode must be select or dense')
        self.early_router = BidirectionalBudgetRouter(**options.pop('router', {}))
        if self.routing_mode == 'dense':
            self.early_router.requires_grad_(False)
        self.image_executor = CoarseTileImageExecutor(**options.pop('image_executor', {}))
        if options:
            raise ValueError('Unknown early budget settings: ' + str(sorted(options)))
        self.last_budget_stats = {}
        self._route_state = None

    def train(self, mode=True):
        super().train(mode)
        # Runner.test() loads a checkpoint without calling init_weights().
        # Enforce the inherited frozen-image policy for standalone evaluation.
        if not mode and self.freeze_img:
            for name in ('img_backbone', 'img_neck'):
                module = getattr(self, name, None)
                if module is not None:
                    module.requires_grad_(False)
        return self

    def extract_feat(self, inputs, metas):
        if self.routing_mode == 'dense':
            self.last_budget_stats = dict(mode='dense')
            return super().extract_feat(inputs, metas)
        imgs = inputs['imgs']
        state = self.early_router(imgs, inputs['voxels'], metas)
        routed = self.early_router.filter_inputs(inputs, state)
        if any(not len(points) for points in routed['points']):
            raise ValueError('MSSF requires at least one in-range radar point per sample')
        img_feats, stats = self.image_executor(
            imgs, state['image_mask'], self.img_backbone,
            self.img_neck if self.with_img_neck else None)
        pts_feats, misc = self.extract_pts_feat(routed['voxels'], points=routed['points'],
            img_feats=img_feats, batch_input_metas=metas, imgs=imgs)
        stats.update(mode='select', radar_pillars_before=len(state['radar_mask']),
                     radar_pillars_after=int(state['radar_mask'].sum()),
                     radar_points_before=sum(len(p) for p in inputs['points']),
                     radar_points_after=sum(len(p) for p in routed['points']))
        self.last_budget_stats = stats
        # Never retain logits/graphs from inference, or from a previous training batch.
        self._route_state = state if self.training and torch.is_grad_enabled() else None
        return img_feats, pts_feats, misc

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        self._route_state = None
        try:
            losses = super().loss(batch_inputs_dict, batch_data_samples, **kwargs)
            if self._route_state is not None:
                losses.update(self.early_router.auxiliary_losses(self._route_state,
                    batch_data_samples, [sample.metainfo for sample in batch_data_samples]))
            return losses
        finally:
            self._route_state = None
