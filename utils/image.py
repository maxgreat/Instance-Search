# -*- encoding: utf-8 -*-

import torchvision.transforms as transforms
from PIL import Image
import cv2
from scipy.ndimage import interpolation
import numpy as np
import random


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
    h_pad = (longer_side - img.size[0]) // 2
    h_mod = (longer_side - img.size[0]) % 2
    v_pad = (longer_side - img.size[1]) // 2
    v_mod = (longer_side - img.size[1]) % 2
    return img.crop((-h_pad - h_mod, -v_pad - v_mod, img.size[0] + h_pad, img.size[1] + v_pad))


# randomly rotate, shift and scale vertically and horizontally a PIL image with given angle in degrees and shifting/scaling ratios
# inspired by http://stackoverflow.com/questions/7501009/affine-transform-in-pil-python
def random_affine(rotation=0, h_range=0, v_range=0, hs_range=0, vs_range=0):
    rotation = rotation * (np.pi / 180)

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
        return im.transform(im.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    return rand_affine


def pad_square_cv(img):
    longer_side = max(img.shape[:2])
    v_pad = (longer_side - img.shape[0]) // 2
    v_mod = (longer_side - img.shape[0]) % 2
    h_pad = (longer_side - img.shape[1]) // 2
    h_mod = (longer_side - img.shape[1]) % 2
    return np.pad(img, ((v_pad + v_mod, v_pad), (h_pad + h_mod, h_pad), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))


def scale_cv(new_size, inter=cv2.INTER_CUBIC):
    if isinstance(new_size, tuple):
        def sc_cv(img):
            return cv2.resize(img, new_size, interpolation=inter)
        return sc_cv
    else:
        def sc_cv(img):
            h, w, _ = img.shape
            if (w <= h and w == new_size) or (h <= w and h == new_size):
                return img
            if w < h:
                ow = new_size
                oh = int(round(float(new_size * h) / w))
                return cv2.resize(img, (ow, oh), interpolation=inter)
            else:
                oh = new_size
                ow = int(round(float(new_size * w) / h))
                return cv2.resize(img, (ow, oh), interpolation=inter)
        return sc_cv


def center_crop_cv(size):
    if not isinstance(size, tuple):
        size = (int(size), int(size))

    def cent_crop_cv(img):
        h, w, _ = img.shape
        th, tw = size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + tw]
    return cent_crop_cv


def random_crop_cv(size):
    if not isinstance(size, tuple):
        size = (int(size), int(size))

    def rand_crop_cv(img):
        h, w, _ = img.shape
        th, tw = size
        if w == tw and h == th:
            return img
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1 + th, x1:x1 + tw]
    return rand_crop_cv


# crop randomly using same aspect ratio as image
# such that shorter side has given size
def random_crop_keep_ar_cv(short_side):
    def rand_crop_cv(img):
        h, w, _ = img.shape
        if (h <= w and h == short_side) or (w <= h and w == short_side):
            return img
        if h < w:
            th = short_side
            tw = int(round(float(short_side * w) / h))
        else:
            tw = short_side
            th = int(round(float(short_side * h) / w))
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1 + th, x1:x1 + tw]
    return rand_crop_cv


def affine_cv(img, angle, v_shift, h_shift, sx, sy, cval=0.):
    # apply translation first to allow the center to be
    # offset to any position when using rotation
    mat = np.array([
        [sy * np.cos(angle), -sy * np.sin(angle), v_shift],
        [sx * np.sin(angle), sx * np.cos(angle), h_shift],
        [0., 0., 1.]
    ])
    # make sure the transform is applied at the center of the image,
    # then reset it afterwards
    offset = (img.shape[0] / 2.0 + 0.5, img.shape[1] / 2.0 + 0.5)
    mat = np.dot(np.dot(
        np.array([
            [1., 0., offset[0]],
            [0., 1., offset[1]],
            [0., 0., 1.]]),
        mat),
        np.array([
            [1., 0., -offset[0]],
            [0., 1., -offset[1]],
            [0., 0., 1.]]))

    def t(channel):
        return interpolation.affine_transform(channel, mat[:2, :2], mat[:2, 2], cval=cval)
    # apply transformation to each channel separately
    return np.dstack(map(t, (img[:, :, i] for i in range(img.shape[2]))))


def random_affine_scale_cv(range_low, range_high):
    def rand_aff_scale_cv(img):
        scale = random.uniform(range_low, range_high)
        return affine_cv(img, 0., 0., 0., scale, scale)
    return rand_aff_scale_cv


def affine_scale_noisy_cv(scale):
    def aff_scale_noisy(img):
        img = affine_cv(img.astype(float), 0., 0., 0., scale, scale, cval=.1)
        img[img == .1] = np.random.randint(256, size=np.sum(img == .1))
        return img.astype(np.uint8)
    return aff_scale_noisy


def random_affine_noisy_cv(rotation=0, h_range=0, v_range=0, hs_range=0, vs_range=0, h_flip=False):
    rotation = rotation * (np.pi / 180)

    def rand_aff_noisy_cv(img):
        # compose the affine transformation applied to x
        angle = np.random.uniform(-rotation, rotation)
        # shift needs to be scaled by size of image in that dimension
        v_shift = np.random.uniform(-v_range, v_range) * img.shape[0]
        h_shift = np.random.uniform(-h_range, h_range) * img.shape[1]
        sx = 1 + random.uniform(-hs_range, hs_range)
        sy = 1 + random.uniform(-vs_range, vs_range)
        if h_flip and random.random() < 0.5:
            sx = -sx
        img = affine_cv(img.astype(float), angle, v_shift, h_shift, sx, sy, cval=.1)
        img[img == .1] = np.random.randint(256, size=np.sum(img == .1))
        return img.astype(np.uint8)
    return rand_aff_noisy_cv


def random_affine_cv(rotation=0, h_range=0, v_range=0, hs_range=0, vs_range=0, h_flip=False):
    rotation = rotation * (np.pi / 180)

    def rand_affine_cv(img):
        # compose the affine transformation applied to x
        angle = np.random.uniform(-rotation, rotation)
        # shift needs to be scaled by size of image in that dimension
        v_shift = np.random.uniform(-v_range, v_range) * img.shape[0]
        h_shift = np.random.uniform(-h_range, h_range) * img.shape[1]
        sx = 1 + random.uniform(-hs_range, hs_range)
        sy = 1 + random.uniform(-vs_range, vs_range)
        if h_flip and random.random() < 0.5:
            sx = -sx
        return affine_cv(img, angle, v_shift, h_shift, sx, sy)
    return rand_affine_cv


def random_h_flip_cv(img):
    return img[:, ::-1, :].copy() if random.random() < 0.5 else img


def imread_rgb(fname):
    # read and convert image from BGR to RGB
    im = cv2.imread(fname)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def tensor_2_bgr(tensor):
    # convert RGB tensor to BGR numpy array as used in OpenCV
    return cv2.cvtColor(tensor.numpy(), cv2.COLOR_RGB2BGR)
