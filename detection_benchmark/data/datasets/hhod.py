import torch
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import ImageFile,Image
import os

from detection_benchmark.structures.bounding_box import BoxList


def get_roots(root):
    return glob.glob(root+'/*/*/*/*/*[0-9].bmp')

data_dir = '/home/razer/Documents/pytorch/handhod_faster_rcnn/datasets/hhod/Raw_trainval'
categorys = os.listdir(data_dir)
categorys = sorted(categorys)
classes_map = {c:i+1 for i,c in enumerate(categorys)}

def get_index_with_classname(name):
    return classes_map[name]

def get_raw_name(root):
    return root.split('/')[-1]

class HHODDataset(Dataset):
    def __init__(
        self, root, ann_file, transforms=None,
    ):
        super(HHODDataset, self).__init__()
        self.root = get_roots(root)
        self.transforms = transforms
        self.raw_name = get_raw_name(root)

    def __getitem__(self, idx):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        img_root = self.root[idx]
        anno = img_root.replace(self.raw_name, 'BBox').replace('bmp', 'txt')
        img = Image.open(img_root).convert('RGB')

        box = np.loadtxt(anno)
        box = torch.as_tensor(box).reshape(1, 4)  # guard against no boxes
        target = BoxList(box, img.size, mode="xywh").convert("xyxy")

        classes = anno.split('/')[-5]
        classes = [get_index_with_classname(classes)]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.root)

    def get_img_info(self, index):
        img_id = self.root[index]
        img_data = Image.open(img_id).size
        return img_data
