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


def _resnet(arch, pretrained2d=False, **kwargs):
    model = ResNet3d(arch, pretrained2d=pretrained2d, **kwargs)
    return model


@registry.BACKBONE.register('resnet18')
def resnet3d_18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(18, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet34')
def resnet3d_34(pretrained=False, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('34', pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet50')
def resnet3d_50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(50, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet101')
def resnet3d_101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(101, pretrained2d=pretrained, **kwargs)


@registry.BACKBONE.register('resnet152')
def resnet3d_152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(152, pretrained2d=pretrained, **kwargs)
