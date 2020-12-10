# -*- coding: utf-8 -*-

"""
@date: 2020/12/7 下午7:58
@file: build_resnet_backbone.py
@author: zj
@description: 
"""

from torchvision.models.utils import load_state_dict_from_url

from .. import registry
from .resnet3d_basicblock import ResNet3DBasicBlock
from .resnet3d_bottleneck import ResNet3DBottleneck
from .resnet3d_backbone import ResNet3DBackbone
from ..norm_helper import get_norm

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

arch_settings = {
    'resnet18': (ResNet3DBasicBlock, (2, 2, 2, 2)),
    'resnet34': (ResNet3DBasicBlock, (3, 4, 6, 3)),
    'resnet50': (ResNet3DBottleneck, (3, 4, 6, 3)),
    'resnet101': (ResNet3DBottleneck, (3, 4, 23, 3)),
    'resnet152': (ResNet3DBottleneck, (3, 8, 36, 3))
}


@registry.BACKBONE.register("ResNet3DBackbone")
def build_resnet3d_backbone(cfg, map_location=None):
    arch = cfg.MODEL.BACKBONE.ARCH
    norm_layer = get_norm(cfg)

    block_layer, layer_blocks = arch_settings[arch]

    torchvision_pretrained = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED
    if torchvision_pretrained:
        state_dict_2d = load_state_dict_from_url(model_urls[arch], progress=True)
    else:
        state_dict_2d = None

    backbone = ResNet3DBackbone(
        base_planes=cfg.MODEL.BACKBONE.BASE_PLANES,
        conv1_kernel=cfg.MODEL.BACKBONE.CONV1_KERNEL,
        conv1_stride=cfg.MODEL.BACKBONE.CONV1_STRIDE,
        conv1_padding=cfg.MODEL.BACKBONE.CONV1_PADDING,
        pool1_kernel=cfg.MODEL.BACKBONE.POOL1_KERNEL,
        pool1_stride=cfg.MODEL.BACKBONE.POOL1_STRIDE,
        pool1_padding=cfg.MODEL.BACKBONE.POOL1_PADDING,
        with_pool2=cfg.MODEL.BACKBONE.WITH_POOL2,
        temporal_strides=cfg.MODEL.BACKBONE.TEMPORAL_STRIDES,
        layer_planes=cfg.MODEL.BACKBONE.LAYER_PLANES,
        layer_blocks=layer_blocks,
        downsamples=cfg.MODEL.BACKBONE.DOWNSAMPLES,
        inflate_list=cfg.MODEL.BACKBONE.INFLATE_LIST,
        inflate_style=cfg.MODEL.BACKBONE.INFLATE_STYLE,
        block_layer=block_layer,
        norm_layer=norm_layer,
        zero_init_residual=cfg.MODEL.BACKBONE.ZERO_INIT_RESIDUAL,
        state_dict_2d=state_dict_2d
    )

    return backbone
