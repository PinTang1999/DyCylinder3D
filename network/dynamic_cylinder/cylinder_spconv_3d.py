# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py
import spconv
from torch import nn

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class DynamicCylinder3D(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 dynamic_segmentor,
                 sparse_shape
                 ):
        super().__init__()
        self.name = "DynamicCylinder3D"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.dynamic_segmentor = dynamic_segmentor

        self.num_stages = len(dynamic_segmentor)

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        """
        @param train_pt_fea_ten: point feature of each point
        @param train_vox_ten: the grid idx
        @param batch_size: batch size
        @param point_wise_label: semantic label of each point
        """
        # coords: sparse voxel [n_voxel, 4], 4: (batch_id, radius, angle, z)
        # features_3d: features of these voxels [n_voxel, C']
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        # out_features [n_voxel, C''], voxel_logits [n_voxel, cls], logits [H, W, D]
        voxel_features, voxel_logits, logits3d, seg_weights = self.cylinder_3d_spconv_seg(features_3d, coords,
                                                                                          batch_size)

        stage_segs = [logits3d]
        for i in range(self.num_stages):
            voxel_logits, seg_weights = self.dynamic_segmentor[i](voxel_features, voxel_logits, seg_weights)
            sparse_logits = spconv.SparseConvTensor(voxel_logits, coords, self.sparse_shape, batch_size)
            stage_segs.append(sparse_logits.dense())

        # stage_segs = stage_segs[:1]
        if self.training:
            return stage_segs

        return stage_segs[-1]
