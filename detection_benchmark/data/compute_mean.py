import glob
import numpy as np
from imageio import imread

data_dir = '/home/razer/Documents/pytorch/handhod_faster_rcnn/datasets/hhod/Raw_trainval'
imgs_path = glob.glob(data_dir+'/*/*/*/*/*[0-9].bmp')


img = imgs_path[0]

R_channel = 0
G_channel = 0
B_channel = 0
num = 0
ws = []
hs = []
for img_path in imgs_path:
    img = imread(img_path)
    w,h = img.shape[:2]
    ws.append(w)
    hs.append(h)
    size = w*h
    num += size
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

ws = np.array(ws)
hs = np.array(hs)
print(np.all(ws==640))
print(np.all(hs==480))
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))