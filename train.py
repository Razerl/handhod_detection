# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

import argparse
import os

import torch
from detection_benchmark.config import cfg
from detection_benchmark.data import make_data_loader
from detection_benchmark.solver import make_lr_scheduler
from detection_benchmark.solver import make_optimizer
from detection_benchmark.engine.inference import inference
from detection_benchmark.engine.trainer import do_train
from detection_benchmark.modeling.detector import build_detection_model
from detection_benchmark.utils.checkpoint import DetectronCheckpointer
from detection_benchmark.utils.collect_env import collect_env_info
from detection_benchmark.utils.comm import synchronize
from detection_benchmark.utils.logging import setup_logger
from detection_benchmark.utils.miscellaneous import mkdir


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    # arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    add_depth = cfg.DATASETS.ADD_DEPTH

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        add_depth,
    )

    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/razer/Documents/pytorch/handhod_faster_rcnn/configs/e2e_mask_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
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
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("detection_benchmark", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
