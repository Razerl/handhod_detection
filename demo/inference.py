from detection_benchmark.config import cfg
from demo.predictor import HHODDemo
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

config_file = "../configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

hhod_demo = HHODDemo(cfg,)
# load image and then run prediction

image = Image.open('01.bmp').convert('RGB')
img = np.asarray(image)

box = np.loadtxt('01.txt')
box = torch.as_tensor(box).reshape(1, 4)  # guard against no boxes

label, score, bbox = hhod_demo.run_image(image, box)
bbox = bbox[0]
cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
txt_center = (bbox[0],bbox[1])
cv2.putText(img,str(label)+':'+str(float('%.4f' %score)),txt_center,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
plt.imshow(img)
plt.show()

