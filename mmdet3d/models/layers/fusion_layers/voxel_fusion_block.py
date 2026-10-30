from typing import List, Union, Optional
import torch
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from .voxel_fusion import proj_to_img

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

    @staticmethod
    def projection_context(points, meta):
        """Validate depth and the unpadded image after the existing augmentations."""
        matrix = get_proj_mat_by_coord_type(meta, 'LIDAR')
        projected = proj_to_img(points, meta, points.new_tensor(matrix),
                               'LIDAR', with_depth=True, normalize=True)
        height, width = meta['input_shape'][:2]
        image_height, image_width = meta.get('img_shape', meta['input_shape'])[:2]
        u, v, depth = projected.unbind(-1)
        valid = (torch.isfinite(projected).all(-1) & (depth > 1e-5)
                 & (u >= 0) & (u < image_width / width)
                 & (v >= 0) & (v < image_height / height))
        geometry = torch.stack((u.clamp(-2, 2), v.clamp(-2, 2),
                                depth.clamp(0, 100).log1p() / 4.6151205168), -1)
        return valid[None, :, None], geometry[None]

    def forward(self, x, voxel_data_dict, img_data_dict, external_query=None):
        # x = voxel_data_dict['sparse_tensor']
        centroids = voxel_data_dict['centroids']
        centroids_img = voxel_data_dict['centroids_img']
        centroids_coor = voxel_data_dict['centroids_coor']
        non_empty_inds = voxel_data_dict['non_empty_inds']
        non_empty_mask = voxel_data_dict['non_empty_mask']
        
        # img_feats = img_data_dict['img_feats']
        feat_flatten = img_data_dict['feat_flatten']
        spatial_shapes = img_data_dict['spatial_shapes']
        level_start_index = img_data_dict['level_start_index']

        # extract feature from image
        non_empty_coor = centroids_coor[non_empty_mask]
        bs = feat_flatten.shape[0]
        # fix bug: batch index is not continuous because we use empty voxel
        #updated_query = []
        #sample_locs_list = []
        updated_query = torch.zeros(x.features[non_empty_inds].shape, dtype=x.features.dtype, device=x.features.device)
        sample_locs_all = torch.zeros([x.features[non_empty_inds].shape[0],4*8*len(level_start_index),2], dtype=x.features.dtype, device=x.features.device)
        sample_weights_all = torch.zeros([x.features[non_empty_inds].shape[0],4*8*len(level_start_index),1], dtype=x.features.dtype, device=x.features.device)
        for i in range(bs):
            batch_mask = (non_empty_coor[:, 0] == i)
            inds = non_empty_inds[batch_mask]
            if not inds.numel():
                continue
            if external_query is not None:
                non_empty_voxel_feats = external_query[batch_mask]
            else:
                non_empty_voxel_feats = x.features[inds]
            ref_pts = centroids_img[batch_mask][:, 1:][None, :, None, :].repeat(1, 1, len(spatial_shapes), 1)
            reliability_context = {}
            if self.img_cross_att.deformable_attention.reliability_gate is not None:
                meta = img_data_dict['batch_input_metas'][i]
                points = centroids[non_empty_mask][batch_mask, 1:4]
                valid, geometry = self.projection_context(points, meta)
                reliability_context = dict(radar_features=x.features[inds][None],
                                           projection_valid=valid, geometry=geometry)
                if getattr(self.img_cross_att.deformable_attention.reliability_gate,
                           'supports_clean_anchor', False):
                    reliability_context['clean_mask'] = torch.full_like(
                        valid, bool(meta.get('v2_is_clean', True)))
            
            ms_feats, sample_locs, sample_weights = self.img_cross_att(
                            query=non_empty_voxel_feats.unsqueeze(0),
                            key=feat_flatten[i].unsqueeze(0),
                            value=feat_flatten[i].unsqueeze(0),
                            reference_points_cams=ref_pts,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            **reliability_context
                        )
            #updated_query[inds] = ms_feats[0]
            #sample_locs_all[inds] = sample_locs[0]
            updated_query[batch_mask] = ms_feats[0]
            sample_locs_all[batch_mask] = sample_locs[0]
            sample_weights_all[batch_mask] = sample_weights[0]

            #updated_query.append(ms_feats[0])
            #sample_locs_list.append(sample_locs)
        return updated_query, sample_locs_all, sample_weights_all
        #return torch.cat(updated_query, dim=0), sample_locs_list
