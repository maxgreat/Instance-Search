# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from os import path
from utils import get_images_labels, match_label_fou_clean2, match_label_video

# resize all images of a dataset and place them into a new folder

# path to folders containing train and test images
dataset = '/home/mrim/data/collection/GUIMUTEIC/FOURVIERE_CLEAN2/TRAIN_I'
dataset_test = '/home/mrim/data/collection/GUIMUTEIC/FOURVIERE_CLEAN2/TEST_I'
# function to match the labels in image names
match_labels = match_label_fou_clean2
# paths where the resized images are placed
out_path = './data/pre_proc/fourviere_clean2_448'
out_path_test = './data/pre_proc/fourviere_clean2_448/test'

# training and test sets (scaled to 300 on the small side)
dataSetFull = get_images_labels(dataset, match_labels)
testSetFull = get_images_labels(dataset_test, match_labels)


# resize function
def resize(dataset, out_path, max_ar, newsize1, newsize2=None):
    for im, lab in dataset:
        im_o = cv2.imread(im)
        h, w, _ = im_o.shape
        if max_ar >= 1. and ((h > w and float(h) / w > max_ar) or (h < w and float(w) / h > max_ar)):
            # force a max aspect ratio of max_ar by padding image with random uniform noise
            def pad_rand(vector, pad_width, iaxis, kwargs):
                if pad_width[0] > 0:
                    vector[:pad_width[0]] = np.random.randint(256, size=pad_width[0])
                if pad_width[1] > 0:
                    vector[-pad_width[1]:] = np.random.randint(256, size=pad_width[1])
                return vector
            if h > w:
                ow = int(np.ceil(float(h) / max_ar))
                w_pad = (ow - w) // 2
                w_mod = (ow - w) % 2
                im_o = np.pad(im_o, ((0, 0), (w_pad + w_mod, w_pad), (0, 0)), pad_rand)
            else:
                oh = int(np.ceil(float(w) / max_ar))
                h_pad = (oh - h) // 2
                h_mod = (oh - h) % 2
                im_o = np.pad(im_o, ((h_pad + h_mod, h_pad), (0, 0), (0, 0)), pad_rand)
        h, w, _ = im_o.shape
        if newsize2 is None:
            if (w <= h and w == newsize1) or (h <= w and h == newsize1):
                ow, oh = w, h
            elif (w < h):
                ow, oh = newsize1, int(round(float(newsize1 * h) / w))
            else:
                ow, oh = int(round(float(newsize1 * w) / h)), newsize1
        else:
            ow, oh = newsize1, newsize2
        if ow == w and oh == h:
            im_out = im_o
        else:
            im_out = cv2.resize(im_o, (ow, oh), interpolation=cv2.INTER_CUBIC)
        out_p = path.join(out_path, im.split('/')[-1])
        print('/'.join(im.split('/')[-3:]), '->', '/'.join(out_p.split('/')[-3:]))
        cv2.imwrite(out_p, im_out)


resize(dataSetFull, out_path, 2.0, 448)
resize(testSetFull, out_path_test, 2.0, 448)
