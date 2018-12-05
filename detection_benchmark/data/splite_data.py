import os
from os.path import join as opj
from detection_benchmark.utils.miscellaneous import mkdir
import shutil
import random
import glob

split = 0.8

data_dir = '/home/razer/Documents/datasets/HHOD/'

raw_dir = opj(data_dir, 'Raw')
train_val_dir = opj(data_dir, 'Raw_trainval')
test_dir = opj(data_dir, 'Raw_test')
mkdir(train_val_dir)
mkdir(test_dir)

for category in os.listdir(raw_dir):
    category_dir = opj(raw_dir, category)
    category_dir_trainval = opj(train_val_dir,category)
    category_dir_test = opj(test_dir,category)
    mkdir(category_dir_trainval)
    mkdir(category_dir_test)
    for instance_num in os.listdir(category_dir):
        instance_dir = opj(category_dir, instance_num)
        instance_dir_trainval = opj(category_dir_trainval, instance_num)
        instance_dir_test = opj(category_dir_test, instance_num)
        mkdir(instance_dir_trainval)
        mkdir(instance_dir_test)
        for people_instance_num in os.listdir(instance_dir):
            people_instance_dir = opj(instance_dir, people_instance_num)
            people_instance_dir_trainval = opj(instance_dir_trainval, people_instance_num)
            people_instance_dir_test = opj(instance_dir_test, people_instance_num)
            mkdir(people_instance_dir_trainval)
            mkdir(people_instance_dir_test)
            for scenes_instance_num in os.listdir(people_instance_dir):
                scenes_instance_dir = opj(people_instance_dir, scenes_instance_num)
                scenes_instance_dir_trainval = opj(people_instance_dir_trainval, scenes_instance_num)
                scenes_instance_dir_test = opj(people_instance_dir_test, scenes_instance_num)
                mkdir(scenes_instance_dir_trainval)
                mkdir(scenes_instance_dir_test)

                imgs_depth = glob.glob(scenes_instance_dir+'/*_d.bmp')
                random.shuffle(imgs_depth)

                num_trainval = int(len(imgs_depth) * split)
                num_test = len(imgs_depth) - num_trainval

                imgs_trainval_depth = imgs_depth[:num_trainval]
                imgs_test_depth = imgs_depth[num_trainval:]
                imgs_trainval_rgb = [i.replace('_d.bmp','.bmp') for i in imgs_trainval_depth]
                imgs_test_rgb = [i.replace('_d.bmp','.bmp') for i in imgs_test_depth]

                for (img_trainval_rgb,img_trainval_depth) in zip(imgs_trainval_rgb,imgs_trainval_depth):
                    shutil.copy(img_trainval_rgb, scenes_instance_dir_trainval)
                    shutil.copy(img_trainval_depth, scenes_instance_dir_trainval)
                for (img_test_rgb,img_test_depth) in zip(imgs_test_rgb,imgs_test_depth):
                    shutil.copy(img_test_rgb, scenes_instance_dir_test)
                    shutil.copy(img_test_depth, scenes_instance_dir_test)
