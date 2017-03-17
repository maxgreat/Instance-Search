# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
import torch
import torchvision.transforms as transforms
import types
import random
import numpy as np
import functools
import itertools
from PIL import Image


# -------------------------- IO stuff --------------------------
def readMeanStd(fname):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


def log_detail(fname, p_file, *args):
    print(*args, file=p_file)
    if fname:
        with open(fname, 'a') as f:
            print(*args, file=f)

def log(fname, *args):
    log_detail(fname, sys.stdout, *args)


# ----------------------- String utils -------------------------
def fun_str(f):
    if f.__class__ in (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType):
        return f.__name__
    else:
        return f.__class__.__name__


def trans_str(trans):
    return ','.join(fun_str(t) for t in trans.transforms)


# ---------------------- Image transformations -----------------
def norm_image_t(tensor):
    m = s = []
    for t in tensor:
        m.append(t.mean())
        s.append(t.std())
    return transforms.Normalize(m, s)(tensor)


# pad a PIL image to a square
def pad_square(img):
    longer_side = max(img.size)
    h_pad = (longer_side - img.size[0]) / 2
    v_pad = (longer_side - img.size[1]) / 2
    return img.crop((-h_pad, -v_pad, img.size[0] + h_pad, img.size[1] + v_pad))


# randomly rotate, shift and scale vertically and horizontally a PIL image with given angle in radians and shifting/scaling ratios
# inspired by http://stackoverflow.com/questions/7501009/affine-transform-in-pil-python
def random_affine(rotation=0, h_range=0, v_range=0, hs_range=0, vs_range=0):
    def rand_affine(im):
        angle = random.uniform(-rotation, rotation)
        x, y = im.size[0] / 2, im.size[1] / 2
        nx = x + random.uniform(-h_range, h_range) * im.size[0]
        ny = y + random.uniform(-v_range, v_range) * im.size[1]
        sx = 1 + random.uniform(-hs_range, hs_range)
        sy = 1 + random.uniform(-vs_range, vs_range)
        cos, sin = np.cos(angle), np.sin(angle)
        a, b = cos / sx, sin / sx
        c = x - nx * a - ny * b
        d, e = -sin / sy, cos / sy
        f = y - nx * d - ny * e
        return im.transform(im.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.NEAREST)
    return rand_affine


# ------------ Generators for couples/triplets ------------------
# get couples of images as a dict with images as keys and all other
# images of same label as values
def get_pos_couples_im(dataset):
    couples = {}
    for x1, l1 in dataset:
        for x2, l2 in dataset:
            if l1 != l2 or x1 == x2:
                continue
            if x1 in couples:
                couples[x1].append(x2)
            else:
                couples[x1] = [x2]


# get the positive couples of a dataset as a dict with labels as keys
def get_pos_couples(dataset):
    couples = {}
    cwr = itertools.combinations_with_replacement
    for (i1, (x1, l1)), (i2, (x2, l2)) in cwr(enumerate(dataset), 2):
        if l1 != l2:
            continue
        t = (l1, (i1, i2), (x1, x2))
        if l1 in couples:
            couples[l1].append(t)
        else:
            couples[l1] = [t]
    return couples


# ----------------------- Other general ------------------------
def tensor_t(t, device, *sizes):
    r = t(*sizes)
    if device >= 0:
        return r.cuda()
    else:
        return r.cpu()


def tensor(device, *sizes):
    return tensor_t(torch.Tensor, device, *sizes)


# evaluate a function by batches of size batch_size on the set x
# and fold over the returned values
def fold_batches(f, init, x, batch_size):
    if batch_size <= 0:
        return f(init, 0, x)
    return functools.reduce(lambda last, idx: f(last, idx, x[idx:min(idx + batch_size, len(x))]), range(0, len(x), batch_size), init)


# ----------------------- Unused ---------------------------------
# get batches of size batch_size from the set x
def batches(x, batch_size):
    for idx in range(0, len(x), batch_size):
        yield x[idx:min(idx + batch_size, len(x))]


def cos_sim(x1, x2, normalized=False):
    if normalized:
        return torch.dot(x1, x2)
    else:
        return torch.dot(x1, x2) / (x1.norm() * x2.norm())
