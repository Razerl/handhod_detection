# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from detection_benchmark.structures.image_list import to_image_list


class BatchCollatorRGBD(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images_rgb = []
        images_d = []
        for t in transposed_batch[0]:
            images_rgb.append(t[0])
            images_d.append(t[1])
        images_rgb = to_image_list(tuple(images_rgb), self.size_divisible)
        images_d = to_image_list(tuple(images_d), self.size_divisible)
        targets = transposed_batch[1]
        return (images_rgb, images_d), targets
