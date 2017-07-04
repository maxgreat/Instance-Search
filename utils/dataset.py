# -*- encoding: utf-8 -*-

import itertools
import glob
import random
import torch
from os import path
from general import tensor_t


def get_images_labels(folder='.', label_f=lambda x: x.split('.')[0]):
    """
        Read a folder containing images where the name of the class is in the filename
        the label function should return the label given the filename
        Return :
            list of couple : (image filename, label)
    """
    exts = ('*.jpg', '*.JPG', '*.JPEG', "*.png")
    r = []
    for ext in exts:
        r.extend([(im, label_f(im)) for im in glob.iglob(path.join(folder, ext))])
    return r


# get couples of images as a dict with images as keys and all
# images of same label as values
def get_pos_couples_ibi(dataset, duplicate=True):
    couples = {}
    for (_, l1, name1), (im2, l2, name2) in itertools.product(dataset, dataset):
        if l1 != l2 or (name1 is name2 and not duplicate):
            continue
        if name1 in couples:
            couples[name1].append(im2)
        else:
            couples[name1] = [im2]
    return couples


# get the positive couples of a dataset as a dict with labels as keys
def get_pos_couples(dataset, duplicate=True):
    couples = {}
    comb = itertools.combinations_with_replacement
    if not duplicate:
        comb = itertools.combinations
    for (i1, (x1, l1, _)), (i2, (x2, l2, _)) in comb(enumerate(dataset), 2):
        if l1 != l2:
            continue
        t = (l1, (i1, i2), (x1, x2))
        if l1 in couples:
            couples[l1].append(t)
        else:
            couples[l1] = [t]
    return couples


# return a random negative for the given label and train set
def choose_rand_neg(train_set, lab):
    im_neg, lab_neg, _ = random.choice(train_set)
    while (lab_neg == lab):
        im_neg, lab_neg, _ = random.choice(train_set)
    return im_neg


# get byte tensors indicating the indexes of images having a different label
def get_lab_indicators(dataset, device):
    n = len(dataset)
    indicators = {}
    for _, lab1, _ in dataset:
        if lab1 in indicators:
            continue
        indicator = tensor_t(torch.ByteTensor, device, n).fill_(0)
        for i2, (_, lab2, _) in enumerate(dataset):
            if lab1 == lab2:
                indicator[i2] = 1
        indicators[lab1] = indicator
    return indicators
