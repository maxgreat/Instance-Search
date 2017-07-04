# -*- encoding: utf-8 -*-

import torch
import torchvision.transforms as transforms
import numpy as np
from utils import get_images_labels, imread_rgb
from utils import match_label_fou_clean2, match_label_video

# create the mean std file needed to normalize images of a dataset

# path to the training images of the dataset
dataset_path = 'data/pre_proc/CLICIDE_video_224sq'
# file to write the mean and std values to
out_path = 'data/CLICIDE_224sq_train_ms.txt'
# function to match labels, this is not necessary here
match_labels = match_label_video
# if the image size is constant, indicate it in format (C, H, W)
# if the image size is not constant, use None here
image_size = (3, 224, 224)
dataset_full = get_images_labels(dataset_path, match_labels)

mean = [0., 0., 0.]
std = [0., 0., 0.]
size = len(dataset_full)
if image_size is not None:
    T = torch.Tensor(size, *(image_size))
    for i, (im, _) in enumerate(dataset_full):
        T[i] = transforms.ToTensor()(imread_rgb(im))
    for i in range(3):
        mean[i] = T[:, i, :, :].mean()
        std[i] = T[:, i, :, :].std()
else:
    # cannot take mean/std of whole dataset tensor.
    # need to compute mean of all pixels and std afterwards, pixel by pixel
    dataset_open = []
    for im, _ in dataset_full:
        im_o = imread_rgb(im) / 255.  # cv2 images are 0-255, torch tensors are 0-1
        im_size = im_o.shape[0] * im_o.shape[1]
        dataset_open.append((im_o, im_size))
        for i in range(3):
            mean[i] += np.sum(im_o[:, :, i]) / (im_size * size)
    for im_o, im_size in dataset_open:
        for i in range(3):
            std[i] += np.sum(np.square(im_o[:, :, i] - mean[i])) / (im_size * size)
    for i in range(3):
        std[i] = np.sqrt(std[i])

with open(out_path, 'w') as outfile:
    outfile.write(' '.join(map(repr, mean)))
    outfile.write('\n')
    outfile.write(' '.join(map(repr, std)))
    outfile.write('\n')
