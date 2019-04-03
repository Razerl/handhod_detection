import torch
import torch.nn.functional as F
import numpy as np


def expand_boxes(boxes, scale_w, scale_h):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale_w
    h_half *= scale_h

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    H = mask.shape[1]
    W = mask.shape[2]

    pad2 = 2 * padding
    scale_w = float(W + pad2) / W
    scale_h = float(H + pad2) / H
    padded_mask = mask.new_zeros((N, 1, H + pad2, W + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale_w, scale_h


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1, is_target=False):
    if is_target:
        scale_h = scale_w = 1.0
    else:
        padded_mask, scale_w, scale_h = expand_masks(mask[None], padding=padding)
        mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale_w, scale_h)[0]
    box = box.to(dtype=torch.int32)

    if not is_target:
        TO_REMOVE = 1
        w = int(box[2] - box[0] + TO_REMOVE)
        h = int(box[3] - box[1] + TO_REMOVE)
        w = max(w, 1)
        h = max(h, 1)

        # Set shape to [batchxCxHxW]
        mask = mask.expand((1, 1, -1, -1))

        # Resize mask
        mask = mask.to(torch.float32)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        mask = mask[0][0]

        if thresh >= 0:
            mask = mask > thresh
        else:
            # for visualization and debugging, we also
            # allow it to return an unmodified mask
            mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


def compute_mask_iou(predict_mask, target_mask):
    predict_mask = predict_mask.numpy()
    target_mask = target_mask.numpy()
    intersection = np.logical_and(predict_mask, target_mask)
    union = np.logical_or(predict_mask, target_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou