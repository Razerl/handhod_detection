# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
from collections import OrderedDict

import torch

from tqdm import tqdm

from detection_benchmark.structures.boxlist_ops import boxlist_iou


def compute_on_dataset(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    right_number = 0
    ious = 0
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            output = model(images, targets)
            output = [o.to(cpu_device) for o in output]
            targets = [t.to(cpu_device) for t in targets]

        for index, (result, target) in enumerate(zip(output, targets)):
            predict_label = result.get_field('labels')[0].to(cpu_device)
            target_label = target.get_field('labels')[0]
            right_number += int(predict_label == target_label)
            iou = boxlist_iou(result,target)[0][0]
            ious += iou
    data_number = len(data_loader.dataset)
    mean_iou = ious/data_number
    accuracy = right_number/data_number
    return (mean_iou, accuracy)


def inference(
    model,
    data_loader,
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
    results = compute_on_dataset(model, data_loader, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    logger.info('mean_iou:{}, accuracy:{}'.format(results[0],results[1]))
    if output_folder:
        torch.save(results, os.path.join(output_folder, "predictions.pth"))
