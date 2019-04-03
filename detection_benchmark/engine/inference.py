# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import torch
from tqdm import tqdm
import numpy as np

from detection_benchmark.structures.boxlist_ops import boxlist_iou
from detection_benchmark.data.datasets.evaluation import paste_mask_in_image, compute_mask_iou


def compute_on_dataset(model, data_loader, device, add_depth):
    model.eval()
    cpu_device = torch.device("cpu")
    right_number = 0
    bbox_ious = 0
    mask_ious = 0
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets = batch
        if add_depth:
            images = (images[0].to(device), images[1].to(device))
        else:
            images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            output = model(images, targets)
            output = [o.to(cpu_device) for o in output]
            targets = [t.to(cpu_device) for t in targets]

        for index, (result, target) in enumerate(zip(output, targets)):
            predict_label = int(result.get_field('labels')[0])
            target_label = int(target.get_field('labels')[0])
            right_number += int(predict_label == target_label)
            bbox_iou = boxlist_iou(result,target)[0][0]
            bbox_ious += bbox_iou
            predict_mask = result.get_field('mask')
            target_mask = target.get_field('masks').polygons[0].convert('mask')
            target_mask = torch.Tensor(np.array(target_mask))
            predict_im_mask = paste_mask_in_image(predict_mask[0,0],result.bbox[0],result.size[1],result.size[0])
            target_im_mask = target_mask
            mask_iou = compute_mask_iou(predict_im_mask,target_im_mask)
            mask_ious += mask_iou

    data_number = len(data_loader.dataset)
    mean_iou_bbox = bbox_ious/data_number
    mean_iou_mask = mask_ious/data_number
    accuracy = right_number/data_number
    return (mean_iou_bbox, mean_iou_mask, accuracy)


def inference(
    model,
    data_loader,
    add_depth,
    device="cuda",
    output_folder=None,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("detection_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    results = compute_on_dataset(model, data_loader, device, add_depth)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    logger.info('mean_iou_bbox:{}, mean_iou_mask:{}, accuracy:{}'.format(results[0], results[1], results[2]))
    if output_folder:
        torch.save(results, os.path.join(output_folder, "predictions.pth"))
