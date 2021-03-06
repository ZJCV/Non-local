# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午7:38
@file: tsn_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from .. import registry


@registry.HEAD.register('ResNet3DHead')
class ResNet3DHead(nn.Module):

    def __init__(self, cfg):
        super(ResNet3DHead, self).__init__()

        in_channels = cfg.MODEL.HEAD.FEATURE_DIMS
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        dropout_rate = cfg.MODEL.HEAD.DROPOUT

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
