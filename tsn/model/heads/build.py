# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from .. import registry
from .tsn_head import TSNHead
from .resnet3d_head import ResNet3DHead

def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)
