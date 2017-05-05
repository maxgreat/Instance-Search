# -*- encoding: utf-8 -*-

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from model.nn_utils import set_net_train
import types
import random
import numpy as np
import functools
import itertools
from PIL import Image
import cv2
from scipy.ndimage import interpolation
import gc


# -------------------------- IO stuff --------------------------
def readMeanStd(fname):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


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


# ------------ Generators for couples/triplets ------------------
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


# ----------------------- Other general ------------------------
def move_device(obj, device):
    if device >= 0:
        return obj.cuda()
    else:
        return obj.cpu()


def tensor_t(t, device, *sizes):
    return move_device(t(*sizes), device)


def tensor(device, *sizes):
    return tensor_t(torch.Tensor, device, *sizes)


# ---------------------- ByteTensor Ops ------------------------
# not operation for a ByteTensor filled with 0, 1
def t_not(t):
    return t.eq(0)


def t_not_(t):
    return t.eq_(0)


# --------------------- General training ------------------------
# evaluate a function by batches of size batch_size on the set x
# and fold over the returned values
def fold_batches(f, init, x, batch_size, cut_end=False, add_args={}):
    nx = len(x)
    if batch_size <= 0:
        return f(init, 0, True, x, **add_args)

    def red(last, idx):
        end = min(idx + batch_size, nx)
        if cut_end and idx + batch_size > nx:
            return last
        is_final = end > nx - batch_size if cut_end else end == nx
        return f(last, idx, is_final, x[idx:end], **add_args)
    return functools.reduce(red, range(0, nx, batch_size), init)


def anneal(net, optimizer, epoch, annealing_dict):
    if epoch not in annealing_dict:
        return optimizer
    default_group = optimizer.state_dict()['param_groups'][0]
    lr = default_group['lr'] * annealing_dict[epoch]
    momentum = default_group['momentum']
    weight_decay = default_group['weight_decay']
    return optim.SGD((p for p in net.parameters() if p.requires_grad), lr=lr, momentum=momentum, weight_decay=weight_decay)


def micro_batch_gen(last, i, is_final, batch, net, create_batch, batch_args, create_loss, loss_avg, loss2_avg, loss2_alpha):
    gc.collect()
    prev_val, mini_batch_size = last
    n = len(batch)
    tensors_in, labels_in = create_batch(batch, n, **batch_args)
    tensors_out = net(*(Variable(t) for t in tensors_in))
    loss, loss2 = create_loss(tensors_out, [Variable(l) for l in labels_in])
    loss_micro = loss * n / mini_batch_size if loss_avg else loss
    val = loss_micro.data[0]
    if loss2 is not None:
        loss2_micro = loss2 * n / mini_batch_size if loss2_avg else loss2
        loss_micro = loss_micro + loss2_alpha * loss2_micro
        val = val + loss2_alpha * loss2_micro.data[0]
    loss_micro.backward()
    return prev_val + val, mini_batch_size


def mini_batch_gen(last, i, is_final, batch, net, optimizer, micro_batch_size, output_stats, stats_args, test_set, epoch, micro_args):
    batch_count, score, running_loss = last
    optimizer.zero_grad()
    loss, _ = fold_batches(micro_batch_gen, (0.0, len(batch)), batch, micro_batch_size, add_args=micro_args)
    optimizer.step()
    running_loss, score = output_stats(net, test_set, epoch, batch_count, is_final, loss, running_loss, score, **stats_args)
    return batch_count + 1, score, running_loss


def train_gen(is_classif, net, train_set, test_set, optimizer, params, create_epoch, create_batch, output_stats, create_loss, best_score=0):
    if is_classif:
        bn_train = params.classif_train_bn
        n_epochs = params.classif_train_epochs
        annealing_dict = params.classif_annealing
        mini_batch_size = params.classif_train_batch_size
        micro_batch_size = params.classif_train_micro_batch
        loss_avg = params.classif_loss_avg
        loss2_avg, loss2_alpha = None, None
    else:
        bn_train = params.siam_train_bn
        n_epochs = params.siam_train_epochs
        annealing_dict = params.siam_annealing
        mini_batch_size = params.siam_train_batch_size
        micro_batch_size = params.siam_train_micro_batch
        loss_avg = params.siam_loss_avg
        loss2_avg = params.siam_do_loss2_avg
        loss2_alpha = params.siam_do_loss2_alpha

    set_net_train(net, True, bn_train=bn_train)
    for epoch in range(n_epochs):
        # annealing
        optimizer = anneal(net, optimizer, epoch, annealing_dict)

        dataset, batch_args, stats_args = create_epoch(epoch, train_set, test_set)

        micro_args = {
            'net': net,
            'create_batch': create_batch,
            'batch_args': batch_args,
            'create_loss': create_loss,
            'loss_avg': loss_avg,
            'loss2_avg': loss2_avg,
            'loss2_alpha': loss2_alpha
        }
        mini_args = {
            'net': net,
            'optimizer': optimizer,
            'micro_batch_size': micro_batch_size,
            'output_stats': output_stats,
            'stats_args': stats_args,
            'test_set': test_set,
            'epoch': epoch,
            'micro_args': micro_args
        }

        init = 0, best_score, 0.0  # batch count, score, running loss
        _, best_score, _ = fold_batches(mini_batch_gen, init, dataset, mini_batch_size, cut_end=True, add_args=mini_args)


# ---------------------- Evaluation metrics -----------------------
# Evaluation metrics (Precision@1 and mAP) given similarity matrix
# Similarity matrix must have size 'test set size' x 'ref set size'
# and contains in each row the similarity of that test (query) image
# with all ref images
def precision1(sim, test_set, ref_set, kth=1):
    total = sim.size(0)
    if kth <= 1:
        max_sim, max_idx = sim.max(1)
    else:
        max_sim, max_idx = sim.kthvalue(sim.size(1) - kth + 1, 1)
    max_label = []
    for i in range(sim.size(0)):
        # get label from ref set which obtained highest score
        max_label.append(ref_set[max_idx[i, 0]][1])
    correct = sum(test_label == max_label[j] for j, (_, test_label, _) in enumerate(test_set))
    return float(correct) / total, correct, total, max_sim, max_label


# according to Oxford buildings dataset definition of AP
# the kth argument allows to ignore the k highest ranked elements of ref set
# this is used to compute AP even for the train set against train set
def avg_precision(sim, i, test_set, ref_set, kth=1):
    test_label = test_set[i][1]
    n_pos = sum(test_label == ref_label for _, ref_label, _ in ref_set)
    n_pos -= (kth - 1)
    if n_pos <= 0:
        return None
    old_recall, old_precision, ap = 0.0, 1.0, 0.0
    intersect_size, j = 0, 0
    _, ranked_list = sim[i].sort(dim=0, descending=True)
    for n, k in enumerate(ranked_list):
        if n + 1 < kth:
            continue
        if ref_set[k][1] == test_label:
            intersect_size += 1

        recall = intersect_size / float(n_pos)
        precision = intersect_size / (j + 1.0)
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
        old_recall, old_precision = recall, precision
        j += 1
    return ap


def mean_avg_precision(sim, test_set, ref_set, kth=1):
    aps = []
    for i in range(sim.size(0)):
        # compute ap for each test image
        ap = avg_precision(sim, i, test_set, ref_set, kth)
        if ap is not None:
            aps.append(ap)
    return sum(aps) / float(len(aps))


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
