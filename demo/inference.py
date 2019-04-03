from demo.predictor import HHODDemo
import numpy as np
import torch
import cv2
from PIL import Image
from detection_benchmark.config import cfg
import os
import random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

config_file = "../configs/e2e_mask_rcnn_R_50_C4_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

hhod_demo = HHODDemo(cfg,)
# load image and then run prediction

# root = './samples'
# images_depth = glob.glob(root+'/*_[a-z].bmp')
# images = [i.replace('_d.bmp', '.bmp') for i in images_depth]
anno_file = '/home/razer/Documents/datasets/HHOD/annotation/instances_test.json'
root = '/home/razer/Documents/datasets/HHOD/Raw_test'
dataset = COCO(anno_file)

for i in range(5):
    image_id = random.randint(24000, 31999)
    ann_ids = dataset.getAnnIds(imgIds=image_id)
    anno = dataset.loadAnns(ann_ids)

    path = dataset.loadImgs(image_id)[0]['file_name'].replace('.bmp', '_d.bmp')
    img = np.array(Image.open(os.path.join(root, path)).convert('RGB'))
    # img = np.asarray(img)
    # img.setflags(write=1)
    plt.figure(0)
    plt.axis('off')
    plt.imshow(img)
    # box_path = image.replace('_d.bmp', '.txt')
    # box = np.loadtxt(box_path)
    box = [obj["bbox"] for obj in anno]
    box = torch.as_tensor(box).reshape(1, 4)  # guard against no boxes

    #show anchors
    # target = BoxList(box,(img.shape[1],img.shape[0]), mode='xywh').convert('xyxy')
    # target.add_field('labels',[1])
    # targets = [target]
    # roi_generator = make_anchor_generator(cfg)
    # proposals = roi_generator(targets)
    # bboxes = proposals[0].bbox.numpy()
    # img_for_anchors = img.copy()
    # for bbox in bboxes:
    #     cv2.rectangle(img_for_anchors,(bbox[0],bbox[1]),(bbox[2],bbox[3]),[255,0,0],4)

    # plt.figure(1)
    # plt.imshow(img_for_anchors)
    # plt.axis('off')

    label, score, bbox, mask = hhod_demo.run_image(img, box)
    bbox = bbox[0]
    img_masked = hhod_demo.composite_img_with_mask(img,mask,label)
    cv2.rectangle(img_masked, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
    txt_center = (bbox[0], bbox[1])
    cv2.putText(img_masked, str(label)+':'+str(float('%.4f' %score)), txt_center,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    # im_name = image.split('.')[1]+'_masked.jpg'
    # im_name = im_name.split('/')[-1]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(im_name, img)
    plt.figure(2)
    plt.imshow(img_masked)
    plt.axis('off')
    plt.show()

# data_loader = make_data_loader(
#         cfg,
#         is_train=True,
#         is_distributed=False,
#         start_iter= 0,
#     )
#
# for iteration, (images, targets) in enumerate(data_loader, 0):
#     if iteration == 1:
#         break
#     for i in range(8):
#         img = targets[i].get_field('masks')[0].polygons[0].convert('mask').numpy()
#         img[img==1] = 255
#         img = Image.fromarray(img, 'L')
#         img.save('mask_{}.jpg'.format(i))
