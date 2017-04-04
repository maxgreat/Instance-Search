# -*- encoding: utf-8 -*-

import torch
import torch.optim as optim
from torch.autograd import Variable
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
# get couples of images as a dict with images as keys and all
# images of same label as values
def get_pos_couples_ibi(dataset, duplicate=True):
    couples = {}
    for (x1, l1), (x2, l2) in itertools.product(dataset, dataset):
        if l1 != l2 or (x1 is x2 and not duplicate):
            continue
        if x1 in couples:
            couples[x1].append(x2)
        else:
            couples[x1] = [x2]
    return couples


# get the positive couples of a dataset as a dict with labels as keys
def get_pos_couples(dataset, duplicate=True):
    couples = {}
    comb = itertools.combinations_with_replacement
    if not duplicate:
        comb = itertools.combinations
    for (i1, (x1, l1)), (i2, (x2, l2)) in comb(enumerate(dataset), 2):
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


# ---------------------- ByteTensor Ops ------------------------
# not operation for a ByteTensor filled with 0, 1
def t_not(t):
    return t.eq(0)


def t_not_(t):
    return t.eq_(0)


# --------------------- General training ------------------------
# evaluate a function by batches of size batch_size on the set x
# and fold over the returned values
def fold_batches(f, init, x, batch_size, cut_end=False):
    nx = len(x)
    if batch_size <= 0:
        return f(init, 0, True, x)

    def red(last, idx):
        end = min(idx + batch_size, nx)
        if cut_end and idx + batch_size > nx:
            return last
        is_final = end > nx - batch_size if cut_end else end == nx
        return f(last, idx, is_final, x[idx:end])
    return functools.reduce(red, range(0, nx, batch_size), init)


def anneal(net, optimizer, epoch, annealing_dict):
    if epoch not in annealing_dict:
        return optimizer
    default_group = optimizer.state_dict()['param_groups'][0]
    lr = default_group['lr'] * annealing_dict[epoch]
    momentum = default_group['momentum']
    weight_decay = default_group['weight_decay']
    return optim.SGD((p for p in net.parameters() if p.requires_grad), lr=lr, momentum=momentum, weight_decay=weight_decay)


def train_gen(is_classif, net, train_set, test_set, criterion, optimizer, params, create_epoch, create_batch, output_stats, loss_choice=None, criterion2=None, loss2_choice=None, best_score=0):
    # do not use double objectives by default
    loss2_alpha, loss2_avg = None, None
    if is_classif:
        n_epochs = params.classif_train_epochs
        annealing_dict = params.classif_annealing
        mini_size = params.classif_train_batch_size
        micro_size = params.classif_train_micro_batch
        loss_avg = params.classif_loss_avg
        if loss_choice is None:
            def loss_choice(outputs, labels):
                return outputs + labels
    else:
        n_epochs = params.siam_train_epochs
        annealing_dict = params.siam_annealing
        mini_size = params.siam_train_batch_size
        micro_size = params.siam_train_micro_batch
        loss_avg = params.siam_loss_avg
        if criterion2:
            loss2_alpha = params.siam_do_loss2_alpha
            loss2_avg = params.siam_do_loss2_avg
        if loss_choice is None:
            def loss_choice(outputs, labels):
                return outputs
    if loss2_choice is None:
        def loss2_choice(outputs, labels):
            return [outputs[0], labels[0]]

    def micro_batch_gen(last, i, is_final, batch):
        prev_loss, mini_batch_size = last
        n = len(batch)
        tensors_in, labels_in = create_batch(batch, n, **batch_args)
        tensors_out = net(*(Variable(t) for t in tensors_in))
        out_list = [tensors_out] if isinstance(tensors_out, Variable) else list(tensors_out)
        loss = criterion(*loss_choice(out_list, [Variable(l) for l in labels_in]))
        loss_micro = loss * n / mini_batch_size
        val = loss_micro.data[0] if loss_avg else loss.data[0]
        if criterion2:
            loss2 = criterion2(*loss2_choice(out_list, [Variable(l) for l in labels_in]))
            loss_micro2 = loss2 * n / mini_batch_size
            loss_micro = loss_micro + loss2_alpha * loss_micro2
            val += loss2_alpha * (loss_micro2.data[0] if loss2_avg else loss2.data[0])
        loss_micro.backward()
        return prev_loss + val, mini_batch_size

    def mini_batch_gen(last, i, is_final, batch):
        batch_count, score, running_loss = last
        optimizer.zero_grad()
        loss, _ = fold_batches(micro_batch_gen, (0.0, len(batch)), batch, micro_size)
        optimizer.step()
        running_loss, score = output_stats(net, test_set, epoch, batch_count, is_final, loss, running_loss, score, **stats_args)
        return batch_count + 1, score, running_loss

    net.train()
    for epoch in range(n_epochs):
        # annealing
        optimizer = anneal(net, optimizer, epoch, annealing_dict)

        dataset, batch_args, stats_args = create_epoch(epoch, train_set, test_set)

        init = 0, best_score, 0.0  # batch count, score, running loss
        _, best_score, _ = fold_batches(mini_batch_gen, init, dataset, mini_size, cut_end=True)


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
    correct = sum(test_label == max_label[j] for j, (_, test_label) in enumerate(test_set))
    return float(correct) / total, correct, total, max_sim, max_label


# according to Oxford buildings dataset definition of AP
# the kth argument allows to ignore the k highest ranked elements of ref set
# this is used to compute AP even for the train set against train set
def avg_precision(sim, i, test_set, ref_set, kth=1):
    test_label = test_set[i][1]
    n_pos = sum(test_label == ref_label for _, ref_label in ref_set)
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
