# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.nn.modules.module import T

from .. import registry
from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..norm_helper import freezing_bn


@registry.RECOGNIZER.register('ResNet3DRecognizer')
class ResNet3DRecognizer(nn.Module):

    def __init__(self, cfg):
        super(ResNet3DRecognizer, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

        self.fix_bn = cfg.MODEL.NORM.FIX_BN
        self.partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    def train(self, mode: bool = True) -> T:
        super(ResNet3DRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, imgs):
        assert len(imgs.shape) == 5

        features = self.backbone(imgs)
        probs = self.head(features)

        return {'probs': probs}
