# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py

import torch.nn as nn
from network.dynamic_cylinder.cylinder_spconv_3d import get_model_class
from network.dynamic_cylinder.segmentator_3d_asymm_spconv import DynamicSparse3D
from network.dynamic_cylinder.cylinder_fea_generator import cylinder_fea
from network.dynamic_cylinder.point_segmentor import PointSegmentor, DynamicPointSegmentor, DynamicSegmentor


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']
    num_stages = model_config['num_stages']

    cylinder_3d_spconv_seg = DynamicSparse3D(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    dynamic_segmentor = nn.ModuleList()
    for i in range(num_stages):
        dynamic_segmentor.append(DynamicSegmentor(feature_channels=4*init_size,
                                                  kernel_channels=4*init_size,
                                                  out_channels=4*init_size,
                                                  num_head=8))

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        dynamic_segmentor=dynamic_segmentor,
        sparse_shape=output_shape
    )

    return model
