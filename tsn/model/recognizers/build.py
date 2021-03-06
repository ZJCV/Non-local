# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午1:58
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

from ..norm_helper import convert_sync_bn
import tsn.util.distributed as du
from tsn.util.checkpoint import CheckPointer
from tsn.util import logging
from .. import registry
from .tsn_recognizer import TSNRecognizer
from .resnet3d_recognizer import ResNet3DRecognizer

logger = logging.get_logger(__name__)


def build_recognizer(cfg, device):
    world_size = du.get_world_size()

    model = registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg).to(device=device)

    if cfg.MODEL.NORM.SYNC_BN and world_size > 1:
        logger.info(
            "start sync BN on the process group of {}".format(du._LOCAL_RANK_GROUP))
        convert_sync_bn(model, du._LOCAL_PROCESS_GROUP)
    if cfg.MODEL.PRETRAINED != "":
        logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=device)
        logger.info("finish loading model weights")

    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model
