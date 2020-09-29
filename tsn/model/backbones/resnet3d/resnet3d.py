# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午4:49
@file: resnet3d.py
@author: zj
@description: 
"""

import torch.nn as nn

from .utility import convTxHxW, _triple, _quadruple
from .basic_block_3d import BasicBlock3d
from .bottleneck_3d import Bottleneck3d


class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(1, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 2.
        with_pool2 (bool): Whether to use pool2. Default: True.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 0, 0).
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_layer (nn.Module): conv layer.
            Default: None.
        norm_layer (nn.Module): norm layers.
            Default: None.
        act_layer (nn.Module): activation layer.
            Default: None.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained2d=True,
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
                 **kwargs):
        super().__init__()
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == len(inflates) == 4
        assert len(conv1_kernel) == 3

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.with_pool2 = with_pool2
        self._make_stem_layer(in_channels, conv1_kernel, conv1_stride_t, pool1_stride_t)

        block, stage_blocks = self.arch_settings[depth]

        self.res_layers = list()
        self.inplanes = 64
        res_planes = [64, 128, 256, 512]
        for i in range(len(stage_blocks)):
            res_layer = self.make_res_layer(block,
                                            res_planes[i],
                                            stage_blocks[i],
                                            spatial_stride=spatial_strides[i],
                                            temporal_stride=temporal_strides[i],
                                            dilation=dilations[i],
                                            inflate=inflates[i],
                                            inflate_style=inflate_style,
                                            conv_layer=self.conv_layer,
                                            norm_layer=self.norm_layer,
                                            act_layer=self.act_layer,
                                            **kwargs
                                            )

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def make_res_layer(self,
                       block,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       norm_layer=None,
                       act_layer=None,
                       conv_layer=None,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            conv_layer (nn.Module): conv layer.
                Default: None.
            norm_layer (nn.Module): norm layers.
                Default: None.
            act_layer (nn.Module): activation layer.
                Default: None.
        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks

        downsample = None
        if spatial_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                convTxHxW(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    padding=0,
                    bias=False,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
                act_layer=act_layer,
                **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer,
                    act_layer=act_layer,
                    **kwargs))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, inplanes, conv1_kernel, conv1_stride_t, pool1_stride_t):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = convTxHxW(
            inplanes,
            64,
            kernel_size=conv1_kernel,
            stride=(conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(conv1_kernel)]),
            bias=False
        )

        self.bn1 = self.norm_layer(64)
        self.relu = self.act_layer(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(pool1_stride_t, 2, 2),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        assert len(x.shape) == 5

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)

        return x