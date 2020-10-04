# -*- coding: utf-8 -*-

"""
@date: 2020/9/25 下午1:56
@file: build_resnet3d.py
@author: zj
@description: 
"""

from torchvision.models.utils import load_state_dict_from_url

from .resnet3d import ResNet3d
from tsn.model import registry

__all__ = ['ResNet3d', 'resnet3d_18', 'resnet3d_34', 'resnet3d_50', 'resnet3d_101',
           'resnet3d_152', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _resnet(arch, cfg, map_location=None):
    pretrained2d = cfg.MODEL.PRETRAINED
    state_dict_2d = None
    if pretrained2d:
        state_dict_2d = load_state_dict_from_url(model_urls[arch],
                                                 progress=True,
                                                 map_location=map_location)

    type = cfg.MODEL.BACKBONE.TYPE
    in_channels = cfg.MODEL.BACKBONE.IN_CHANNELS
    spatial_strides = cfg.MODEL.BACKBONE.SPATIAL_STRIDES
    temporal_strides = cfg.MODEL.BACKBONE.TEMPORAL_STRIDES
    dilations = cfg.MODEL.BACKBONE.DILATIONS
    conv1_kernel = cfg.MODEL.BACKBONE.CONV1_KERNEL
    conv1_stride_t = cfg.MODEL.BACKBONE.CONV1_STRIDE_T
    pool1_stride_t = cfg.MODEL.BACKBONE.POOL1_STRIDE_T
    with_pool2 = cfg.MODEL.BACKBONE.WITH_POOL2
    inflates = cfg.MODEL.BACKBONE.INFLATES
    inflate_style = cfg.MODEL.BACKBONE.INFLATE_STYLE
    if type == 'C2D':
        model = ResNet3d(arch,
                         in_channels=in_channels,
                         spatial_strides=spatial_strides,
                         temporal_strides=temporal_strides,
                         dilations=dilations,
                         conv1_kernel=conv1_kernel,
                         conv1_stride_t=conv1_stride_t,
                         pool1_stride_t=pool1_stride_t,
                         with_pool2=with_pool2,
                         inflates=inflates,
                         inflate_style=inflate_style,
                         zero_init_residual=True,
                         state_dict_2d=state_dict_2d)
    elif type == 'I3D_3x3x3':
        model = ResNet3d(arch,
                         in_channels=3,
                         spatial_strides=(1, 2, 2, 2),
                         temporal_strides=(1, 1, 1, 1),
                         dilations=(1, 1, 1, 1),
                         conv1_kernel=(5, 7, 7),
                         conv1_stride_t=2,
                         pool1_stride_t=2,
                         with_pool2=True,
                         inflates=(1, 1, 1, 1),
                         inflate_style='3x3x3',
                         zero_init_residual=True,
                         state_dict_2d=state_dict_2d)
    elif type == 'I3D_3x1x1':
        model = ResNet3d(arch,
                         in_channels=3,
                         spatial_strides=(1, 2, 2, 2),
                         temporal_strides=(1, 1, 1, 1),
                         dilations=(1, 1, 1, 1),
                         conv1_kernel=(5, 7, 7),
                         conv1_stride_t=2,
                         pool1_stride_t=2,
                         with_pool2=True,
                         inflates=(1, 1, 1, 1),
                         inflate_style='3x1x1',
                         zero_init_residual=True,
                         state_dict_2d=state_dict_2d)
    else:
        raise ValueError('no matching type')
    return model


@registry.BACKBONE.register('resnet3d_18')
def resnet3d_18(type='C2D', pretrained=False, **kwargs):
    return _resnet("resnet18", type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet3d_34')
def resnet3d_34(type='C2D', pretrained=False, **kwargs):
    return _resnet("resnet34", type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet3d_50')
def resnet3d_50(cfg, map_location=None):
    return _resnet("resnet50", cfg, map_location=None)


@registry.BACKBONE.register('resnet3d_101')
def resnet3d_101(type='C2D', pretrained=False, **kwargs):
    return _resnet("resnet101", type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet3d_152')
def resnet3d_152(type='C2D', pretrained=False, **kwargs):
    return _resnet("resnet152", type, pretrained2d=pretrained, **kwargs)
