# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
import random
import numpy as np

from detection_benchmark.modeling.detector import build_detection_model
from detection_benchmark.utils.checkpoint import DetectronCheckpointer
from detection_benchmark.structures.image_list import to_image_list
from detection_benchmark.data.transforms import build_transforms
from detection_benchmark.structures.bounding_box import BoxList
from detection_benchmark.data.datasets.evaluation import paste_mask_in_image


class HHODDemo(object):
    # HHOD categories for pretty print
    CATEGORIES = [
        "__background",
        "apple",
        "bowl",
        "ball",
        "bottle",
        "banana",
        "box",
        "bat",
        "book",
        "calendar",
        "cucumber",
        "cup",
        "carrot",
        "can",
        "calculator",
        "disk",
        "dish",
        "fan",
        "glove",
        "hat",
        "handbag",
        "instant_noodle_bag",
        "kiwi",
        "keyboard",
        "mobilephone",
        "melon",
        "mouse",
        "medicine_bottle",
        "mobileHDD",
        "neck pillow",
        "orange",
        "paper_cup",
        "pitaya",
        "socket",
        "stapler",
        "toothpaste",
        "tomato",
        "towel",
        "tissue",
        "trashcan",
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
        self.labels_color = {}
        for label in self.CATEGORIES:
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            self.labels_color[label] = color

    def composite_img_with_mask(self, img, mask, label):
        x, y = np.where(mask == 1)
        color = np.array(self.labels_color[label])
        for xi, yi in zip(x, y):
            img[xi, yi] = color
        return img

    def run_image(self, image, box):
        target = BoxList(box, image.shape[:2], mode="xywh").convert("xyxy")
        target.add_field('labels',[])
        image, target = self.transforms(image, target)

        image = (image,)
        target = (target,)
        target = [t.to(self.device) for t in target]
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_list, target)
            predictions = [p.to(self.cpu_device) for p in predictions]
            prediction = predictions[0]

        label = prediction.get_field('labels')[0].to(self.cpu_device)
        label = self.CATEGORIES[label]
        score = prediction.get_field('scores')[0].to(self.cpu_device)
        bbox = prediction.bbox
        mask = prediction.get_field('mask')
        im_mask = paste_mask_in_image(mask[0,0],bbox[0],prediction.size[1],prediction.size[0])
        return label, score, bbox,im_mask.numpy()
