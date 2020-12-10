# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()
    _C.MODEL.NAME = "TSN"
    _C.MODEL.PRETRAINED = ""

    _C.MODEL.NORM = CN()
    _C.MODEL.NORM.TYPE = 'BatchNorm2d'
    # for bn
    _C.MODEL.NORM.SYNC_BN = False
    _C.MODEL.NORM.FIX_BN = False
    _C.MODEL.NORM.PARTIAL_BN = False
    # for groupnorm
    _C.MODEL.NORM.GROUPS = 32

    _C.MODEL.ACT = CN()
    _C.MODEL.ACT.TYPE = 'ReLU'

    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = 'ResNetBackbone'
    _C.MODEL.BACKBONE.TORCHVISION_PRETRAINED = False
    # for ResNet
    _C.MODEL.BACKBONE.ARCH = 'resnet18'
    _C.MODEL.BACKBONE.ZERO_INIT_RESIDUAL = False
    _C.MODEL.BACKBONE.BASE_PLANES = 64
    _C.MODEL.BACKBONE.LAYER_PLANES = (64, 128, 256, 512)
    _C.MODEL.BACKBONE.DOWNSAMPLES = (0, 1, 1, 1)
    # for ResNet3D
    _C.MODEL.BACKBONE.CONV1_KERNEL = (1, 7, 7)
    _C.MODEL.BACKBONE.CONV1_STRIDE = (1, 2, 2)
    _C.MODEL.BACKBONE.CONV1_PADDING = (0, 3, 3)
    _C.MODEL.BACKBONE.POOL1_KERNEL = (1, 3, 3)
    _C.MODEL.BACKBONE.POOL1_STRIDE = (1, 2, 2)
    _C.MODEL.BACKBONE.POOL1_PADDING = (0, 1, 1)
    _C.MODEL.BACKBONE.WITH_POOL2 = False
    _C.MODEL.BACKBONE.TEMPORAL_STRIDES = (1, 1, 1, 1)
    _C.MODEL.BACKBONE.INFLATE_LIST = (0, 0, 0, 0)
    _C.MODEL.BACKBONE.INFLATE_STYLE = '3x1x1'

    _C.MODEL.HEAD = CN()
    _C.MODEL.HEAD.NAME = 'TSNHead'
    _C.MODEL.HEAD.FEATURE_DIMS = 2048
    _C.MODEL.HEAD.DROPOUT = 0.0
    _C.MODEL.HEAD.NUM_CLASSES = 51

    _C.MODEL.RECOGNIZER = CN()
    _C.MODEL.RECOGNIZER.NAME = 'TSNRecognizer'

    _C.MODEL.CONSENSU = CN()
    _C.MODEL.CONSENSU.NAME = 'AvgConsensus'

    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'
