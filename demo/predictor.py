# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from detection_benchmark.modeling.detector import build_detection_model
from detection_benchmark.utils.checkpoint import DetectronCheckpointer
from detection_benchmark.structures.image_list import to_image_list
from detection_benchmark.data.transforms import build_transforms
from detection_benchmark.structures.bounding_box import BoxList


class HHODDemo(object):
    # HHOD categories for pretty print
    CATEGORIES = [
        "__background",
        "apple",
        "ball",
        "banana",
        "bat",
        "book",
        "bottle",
        "bowl",
        "box",
        "calculator",
        "calendar",
        "can",
        "carrot",
        "cucumber",
        "cup",
        "dish",
        "disk",
        "fan",
        "glove",
        "handbag",
        "hat",
        "instant_noodle_bag",
        "keyboard",
        "kiwi",
        "medicine_bottle",
        "melon",
        "mobileHDD",
        "mobilephone",
        "mouse",
        "neck pillow",
        "orange",
        "pitaya",
        "socket",
        "stapler",
        "tissue",
        "tomato",
        "toothpaste",
        "towel",
        "trashcan",
        "vegetation",
        "zongzi",
    ]

    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        checkpointer = DetectronCheckpointer(cfg, self.model)

        _ = checkpointer.load('../'+cfg.MODEL.WEIGHT)

        self.transforms = build_transforms(self.cfg, is_train=False)

        self.cpu_device = torch.device("cpu")

    def run_image(self, image, box):
        target = BoxList(box, image.size, mode="xywh").convert("xyxy")
        target.add_field('labels',[])
        image, target = self.transforms(image, target)

        image = (image,)
        target = (target,)
        target = [t.to(self.device) for t in target]
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_list,target)
            predictions = [p.to(self.cpu_device) for p in predictions]
            prediction = predictions[0]

        label = prediction.get_field('labels')[0].to(self.cpu_device)
        label = self.CATEGORIES[label]
        score = prediction.get_field('scores')[0].to(self.cpu_device)
        bbox = prediction.bbox
        return label, score, bbox
