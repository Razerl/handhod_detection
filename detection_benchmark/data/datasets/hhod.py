import torch
import glob
import os
import torchvision
from detection_benchmark.structures.segmentation_mask import SegmentationMask
from detection_benchmark.structures.bounding_box import BoxList
from PIL import Image


class HHODDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, root, ann_file, remove_images_without_annotations, add_depth, transforms=None,
    ):
        super(HHODDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.add_depth = add_depth

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms

    def __getitem__(self, idx):
        # img, anno = super(HHODDataset, self).__getitem__(idx)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name'].replace('.bmp', '_d.bmp')

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.add_depth:
            coco = self.coco
            img_id = self.ids[idx]
            path = coco.loadImgs(img_id)[0]['file_name'].replace('.bmp', '_d.bmp')
            img_depth = Image.open(os.path.join(self.root, path))

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = torch.Tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.add_depth:
            if self.transforms is not None:
                img_depth, _ = self.transforms(img_depth, target)
            return (img, img_depth), target, idx
        else:
            return img, target, idx

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_data = self.coco.imgs[img_id]
        return img_data
