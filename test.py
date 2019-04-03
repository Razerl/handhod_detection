# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported

import argparse
import os

import torch
from detection_benchmark.config import cfg
from detection_benchmark.data import make_data_loader
from detection_benchmark.engine.inference import inference
from detection_benchmark.modeling.detector import build_detection_model
from detection_benchmark.utils.checkpoint import DetectronCheckpointer
from detection_benchmark.utils.collect_env import collect_env_info
from detection_benchmark.utils.comm import synchronize
from detection_benchmark.utils.logging import setup_logger
from detection_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/home/razer/Documents/pytorch/handhod_faster_rcnn/configs/e2e_mask_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("detection_benchmark", save_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    add_depth = cfg.DATASETS.ADD_DEPTH
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            add_depth,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,

        )
        synchronize()


if __name__ == "__main__":
    main()
