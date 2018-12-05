# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from . import resnet


def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


_BACKBONES = {"resnet": build_resnet_backbone}


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY.startswith(
        "R-"
    ), "Only ResNet and ResNeXt models are currently implemented"
    # Models using FPN end with "-FPN"
    if cfg.MODEL.BACKBONE.CONV_BODY.endswith("-FPN"):
        print('sorry this model have not build fpn')
    return build_resnet_backbone(cfg)
