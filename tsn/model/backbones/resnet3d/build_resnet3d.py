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


def _resnet(arch, type, pretrained2d=False, **kwargs):
    if type == 'C2D':
        model = ResNet3d(arch,
                         pretrained2d=pretrained2d,
                         in_channels=3,
                         spatial_strides=(1, 2, 2, 2),
                         temporal_strides=(1, 1, 1, 1),
                         dilations=(1, 1, 1, 1),
                         conv1_kernel=(1, 7, 7),
                         conv1_stride_t=2,
                         pool1_stride_t=2,
                         with_pool2=True,
                         inflates=(0, 0, 0, 0),
                         inflate_style='3x1x1',
                         conv_layer=None,
                         norm_layer=None,
                         act_layer=None,
                         zero_init_residual=True,
                         **kwargs)
    elif type == 'I3D_3x3x3':
        model = ResNet3d(arch,
                         pretrained2d=pretrained2d,
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
                         conv_layer=None,
                         norm_layer=None,
                         act_layer=None,
                         zero_init_residual=True,
                         **kwargs)
    elif type == 'I3D_3x1x1':
        model = ResNet3d(arch,
                         pretrained2d=pretrained2d,
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
                         conv_layer=None,
                         norm_layer=None,
                         act_layer=None,
                         zero_init_residual=True,
                         **kwargs)
    else:
        raise ValueError('no matching type')
    return model


@registry.BACKBONE.register('resnet18')
def resnet3d_18(type='C2D', pretrained=False, **kwargs):
    return _resnet(18, type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet34')
def resnet3d_34(type='C2D', pretrained=False, **kwargs):
    return _resnet(34, type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet50')
def resnet3d_50(type='C2D', pretrained=False, progress=True, **kwargs):
    return _resnet(50, type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet101')
def resnet3d_101(type='C2D', pretrained=False, **kwargs):
    return _resnet(101, type, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet152')
def resnet3d_152(type='C2D', pretrained=False, **kwargs):
    return _resnet(152, type, pretrained2d=pretrained, **kwargs)
