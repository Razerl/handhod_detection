# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn
from detection_benchmark.structures.bounding_box import BoxList
from detection_benchmark.structures.boxlist_ops import boxlist_iou


class GuidedAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        anchor_scale=8,
        anchor_strides=(16,),
        add_depth=False,
        anchor_num=9,
        in_channel=512,
        straddle_thresh=0,
    ):
        super(GuidedAnchorGenerator, self).__init__()

        assert len(anchor_strides) == 1, "we only support one stride for now"
        anchor_stride = anchor_strides[0]
        anchor_num = anchor_num * (int(add_depth)+1)
        self.strides = anchor_stride
        self.scale = anchor_scale
        self.anchor_num = anchor_num
        self.add_depth = add_depth
        self.straddle_thresh = straddle_thresh

        self.predict_wh = nn.Conv2d(in_channel, 2, kernel_size=1, stride=1)
        for l in [self.predict_wh, ]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, features, targets):
        """

        :param features: List(features_rgb, features_d)
        :param targets:  List(BoxList)
        :return: anchors
        """
        num_features = int(self.add_depth)+1
        assert len(features) == num_features, 'if add_depth, there should be 2 features'
        N, C, H, W = features[0].shape
        device = features[0].device
        ctrs = []
        index_anchors = []
        for i, target in enumerate(targets):
            for bbox in target.bbox:
                _, _, x_ctr, y_ctr = _whctrs(bbox)
                ctrs.append(torch.cat((y_ctr.view(-1), x_ctr.view(-1))))
                x_ctr_scaled, y_ctr_scaled = x_ctr/self.strides, y_ctr/self.strides
                index_anchor = get_index_anchors(W, (x_ctr_scaled, y_ctr_scaled))
                index_anchors.append(index_anchor)

        ctrs = torch.cat(ctrs).repeat(1, self.anchor_num).view(N * self.anchor_num, 2)
        whs = []
        batch_idx = torch.arange(N, device=device)[:, None]
        index_anchors = torch.cat(index_anchors).long().view(N, -1)
        for feature in features:
            predict_wh = self.predict_wh(feature)
            predict_wh = predict_wh.permute(0, 2, 3, 1).view(N, W*H, 2)
            anchors_wh = predict_wh[batch_idx, index_anchors]
            whs.append(anchors_wh)
        whs = torch.cat(whs).view(num_features, N, -1, 2).permute(1, 2, 0, 3).view(N * self.anchor_num, 2)
        whs = transfer_whs(self.scale, self.strides, whs)
        anchors = torch.cat((ctrs, whs), 1)
        anchors = _mkanchors(anchors).view(N, self.anchor_num, 4)
        anchors_no_grad = anchors.clone().detach()
        rois = []
        rois_no_grad = []
        for i in range(N):
            label = targets[i].get_field('labels')
            roi = BoxList(anchors[i], targets[i].size, targets[i].mode)
            roi_no_grad = BoxList(anchors_no_grad[i], targets[i].size, targets[i].mode)
            # label = label.repeat(len(roi))
            label = len(roi) * [label]
            roi_no_grad.add_field('labels', label)
            rois.append(roi)
            rois_no_grad.append(roi_no_grad)
        anchor_loss = self.compute_loss(rois, targets, device=device)
        loss = {"loss_anchor_iou": anchor_loss}
        rois = self.remove_visibility(rois_no_grad)
        return rois, loss

    def compute_loss(self, anchors, targets, device):
        loss = 0
        for i in range(len(anchors)):
            iou = boxlist_iou(anchors[i], targets[i])
            loss += torch.sum(1-iou)
        return loss / (len(anchors) * self.anchor_num)

    def remove_visibility(self, boxlists):
        results = []
        for boxlist in boxlists:
            image_width, image_height = boxlist.size
            anchors = boxlist.bbox
            anchors[:, 0] = torch.clamp(anchors[:, 0], min=self.straddle_thresh, max=image_width + self.straddle_thresh)
            anchors[:, 1] = torch.clamp(anchors[:, 1], min=self.straddle_thresh, max=image_height + self.straddle_thresh)
            anchors[:, 2] = torch.clamp(anchors[:, 2], min=self.straddle_thresh, max=image_width + self.straddle_thresh)
            anchors[:, 3] = torch.clamp(anchors[:, 3], min=self.straddle_thresh, max=image_height + self.straddle_thresh)
            # anchors[:, 0][anchors[:, 0] < self.straddle_thresh] = self.straddle_thresh
            # anchors[:, 1][anchors[:, 1] < self.straddle_thresh] = self.straddle_thresh
            # anchors[:, 2][anchors[:, 2] >= image_width + self.straddle_thresh] = image_width + self.straddle_thresh
            # anchors[:, 3][anchors[:, 3] >= image_height + self.straddle_thresh] = image_height + self.straddle_thresh
            boxlist.bbox = anchors
            results.append(boxlist)
        return results


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    anchor_scale = config.MODEL.RPN.ANCHOR_SCALE
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    anchor_num = config.MODEL.RPN.ANCHOR_NUM
    add_depth = config.DATASETS.ADD_DEPTH
    in_channel = config.MODEL.BACKBONE.OUT_CHANNELS
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = GuidedAnchorGenerator(
        anchor_scale, anchor_stride, add_depth, anchor_num, in_channel, straddle_thresh
    )
    return anchor_generator


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(anchors):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    anchors[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)
    anchors[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)
    anchors[:, 2] = anchors[:, 0] + 0.5 * (anchors[:, 2] - 1)
    anchors[:, 3] = anchors[:, 1] + 0.5 * (anchors[:, 3] - 1)

    return anchors


def transfer_whs(scale, stride, whs):
    return scale * stride * torch.exp(whs)


def get_index_anchors(W, ctr):
    x_ctr, y_ctr = ctr
    index = []
    for i in range(-2, 1):
        for j in range(-2,1):
            ind = (W*(x_ctr+i)+y_ctr+j)
            index.append(ind.view(-1))
    index = torch.cat(index)
    return index
