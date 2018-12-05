# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn
from detection_benchmark.structures.bounding_box import BoxList


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(16,),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        assert len(anchor_strides) == 1, "we only support one stride for now"
        anchor_stride = anchor_strides[0]
        anchors, x_ctr, y_ctr = generate_anchors(anchor_stride, sizes, aspect_ratios)
        self.base_x_ctr = x_ctr
        self.base_y_ctr = y_ctr

        self.strides = anchor_stride
        self.cell_anchors = anchors
        self.straddle_thresh = straddle_thresh

    def remove_visibility(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        anchors[..., 0][anchors[..., 0] < -self.straddle_thresh] = self.straddle_thresh
        anchors[..., 1][anchors[..., 1] < -self.straddle_thresh] = self.straddle_thresh
        anchors[..., 2][anchors[..., 2] >= image_width + self.straddle_thresh] = image_width + self.straddle_thresh
        anchors[..., 3][anchors[..., 3] >= image_height + self.straddle_thresh] = image_height + self.straddle_thresh
        boxlist.bbox = anchors

    def forward(self, targets):
        anchors = []
        for target in targets:
            for bbox in target.bbox:
                _, _, x_ctr, y_ctr = _whctrs(bbox)
                shift_x = x_ctr-self.base_x_ctr
                shift_y = y_ctr-self.base_y_ctr

                cell_anchors = self.cell_anchors.clone()
                cell_anchors[:, 0] = cell_anchors[:,0]+shift_x
                cell_anchors[:, 2] = cell_anchors[:,2]+shift_x
                cell_anchors[:, 1] = cell_anchors[:,1]+shift_y
                cell_anchors[:, 3] = cell_anchors[:,3]+shift_y
                anchors.append(cell_anchors)

        rois = []
        for index_anchors_per_image in range(len(anchors)):
            anchor = anchors[index_anchors_per_image].to(targets[index_anchors_per_image].bbox.device)
            roi = BoxList(anchor, targets[index_anchors_per_image].size,
                          targets[index_anchors_per_image].mode)
            self.remove_visibility(roi)
            label = targets[index_anchors_per_image].get_field('labels')
            roi.add_field('labels', label)
            rois.append(roi)
        return rois


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )
    return anchor_generator


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors, x_ctr, y_ctr = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors), x_ctr, y_ctr


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors,x_ctr,y_ctr


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

