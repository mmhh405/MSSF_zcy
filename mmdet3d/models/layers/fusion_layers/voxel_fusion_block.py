from typing import List, Union, Optional
import torch
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS

@MODELS.register_module()
class VoxelFusionBlock(BaseModule):
    def __init__(self,
                 img_cross_att: Optional[dict] = 
                    dict(type='RadarImageCrossAttention',
                        query_embed_dims=256,
                        value_embed_dims=256,
                        output_embed_dims=256,
                        deformable_attention=dict(
                            type='MSDeformableAttention',
                            num_levels=4
                            ),
                    ),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.img_cross_att = MODELS.build(img_cross_att)

    def forward(self, x, voxel_data_dict, img_data_dict, external_query=None,
                return_sampling=True):
        centroids_coor = voxel_data_dict['centroids_coor']
        non_empty_inds = voxel_data_dict['non_empty_inds']
        non_empty_mask = voxel_data_dict['non_empty_mask']
        
        feat_flatten = img_data_dict['feat_flatten']
        spatial_shapes = img_data_dict['spatial_shapes']
        level_start_index = img_data_dict['level_start_index']

        non_empty_coor = centroids_coor[non_empty_mask]
        centroids_img = voxel_data_dict['centroids_img'][non_empty_mask]
        camera_depth = voxel_data_dict.get('camera_depth')
        if camera_depth is not None:
            camera_depth = camera_depth[non_empty_mask]
        bs = feat_flatten.shape[0]
        num_queries = non_empty_inds.numel()
        updated_query = x.features.new_zeros(
            (num_queries, x.features.shape[1]))
        sample_locs_all = None
        sample_weights_all = None
        if return_sampling:
            num_samples = 4 * 8 * len(level_start_index)
            sample_locs_all = x.features.new_zeros((num_queries, num_samples, 2))
            sample_weights_all = x.features.new_zeros((num_queries, num_samples, 1))
        for i in range(bs):
            batch_mask = (non_empty_coor[:, 0] == i)
            inds = non_empty_inds[batch_mask]
            if inds.numel() == 0:
                continue
            if external_query is not None:
                non_empty_voxel_feats = external_query[batch_mask]
            else:
                non_empty_voxel_feats = x.features[inds]
            ref_pts = centroids_img[batch_mask][:, 1:][None, :, None, :].expand(-1, -1, len(spatial_shapes), -1)
            
            prior_kwargs = {}
            if camera_depth is not None:
                depth = camera_depth[batch_mask]
                # Approximate the image footprint of a fixed metric offset.
                # Missing radar support (zero depth) uses the original window.
                valid = torch.isfinite(depth) & (depth > 0)
                scale = (20.0 / depth.clamp_min(1.0)).clamp(0.5, 2.0)
                scale = torch.where(valid, scale, 1.0)
                prior_kwargs['sampling_scale'] = scale.unsqueeze(0).to(
                    non_empty_voxel_feats.dtype)

            ms_feats, sample_locs, sample_weights = self.img_cross_att(
                            query=non_empty_voxel_feats.unsqueeze(0),
                            key=feat_flatten[i].unsqueeze(0),
                            value=feat_flatten[i].unsqueeze(0),
                            reference_points_cams=ref_pts,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            **prior_kwargs,
                        )
            updated_query[batch_mask] = ms_feats[0]
            if return_sampling:
                sample_locs_all[batch_mask] = sample_locs[0]
                sample_weights_all[batch_mask] = sample_weights[0]

        return updated_query, sample_locs_all, sample_weights_all
