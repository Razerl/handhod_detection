# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from detection_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from detection_benchmark.modeling.anchor_generator.anchor_generator import make_anchor_generator
# from detection_benchmark.modeling.anchor_generator.guided_anchor_generator import make_anchor_generator


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = make_anchor_generator(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.add_depth = cfg.DATASETS.ADD_DEPTH

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.add_depth:
            images_rgb, images_d = images
            images_rgb = to_image_list(images_rgb)
            images_d = to_image_list(images_d)
            features = []
            features_rgb = self.backbone(images_rgb.tensors)
            features_d = self.backbone(images_d.tensors)
            features.append(features_rgb[0])
            features.append(features_d[0])
        else:
            images = to_image_list(images)
            features = self.backbone(images.tensors)
        proposals = self.rpn(targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            # losses.update(proposal_loss)
            return losses

        return result
