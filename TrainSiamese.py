# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

import math
import itertools
import functools
import random
import time
from os import path, rename
import tempfile
from uuid import uuid1

from PIL import Image

from model.siamese import *
from dataset import ReadImages

# TODO create generator to yield couples of images
# / triplets (need a way to identify positive couples for each images,
# then iterate over all others to create triples)


def trans_str(trans):
    return ','.join(str(t) for t in trans.transforms)


def readMeanStd(fname):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


# pad a PIL image to a square
def pad_square(img):
    longer_side = max(img.size)
    h_pad = (longer_side - img.size[0]) / 2
    v_pad = (longer_side - img.size[1]) / 2
    return img.crop((-h_pad, -v_pad, img.size[0] + h_pad, img.size[1] + v_pad))


# randomly rotate, shift and scale vertically and horizontally a PIL image with given angle in radians and shifting/scaling ratios
# inspired by http://stackoverflow.com/questions/7501009/affine-transform-in-pil-python
def random_affine(rotation=0, h_range=0, v_range=0, hs_range=0, vs_range=0):
    def trans(im):
        angle = random.uniform(-rotation, rotation)
        x, y = im.size[0] / 2, im.size[1] / 2
        nx = x + random.uniform(-h_range, h_range) * im.size[0]
        ny = y + random.uniform(-v_range, v_range) * im.size[1]
        sx = 1 + random.uniform(-hs_range, hs_range)
        sy = 1 + random.uniform(-vs_range, vs_range)
        cos, sin = math.cos(angle), math.sin(angle)
        a, b = cos / sx, sin / sx
        c = x - nx * a - ny * b
        d, e = -sin / sy, cos / sy
        f = y - nx * d - ny * e
        return im.transform(im.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.NEAREST)
    return trans


class TestParams(object):

    def __init__(self):
        # UUID for these parameters (at random)
        self.uuid = uuid1()

        # general parameters
        self.dataset_full = 'data/pre_proc/CLICIDE_227sq'
        self.dataset_name = self.dataset_full.split('/')[-1].split('_')[0]
        self.mean_std_file = 'data/cli.txt' if self.dataset_name == 'CLICIDE' else 'data/fou.txt'
        self.finetuning = True
        self.save_dir = 'data'
        self.cuda_device = 1

        # in ResNet, before first layer, there are 2 modules with parameters.
        # then number of blocks per layers:
        # ResNet152 - layer 1: 3, layer 2: 8, layer 3: 36, layer 4: 3
        # ResNet50 - layer 1: 3, layer 2: 4, layer 3: 6, layer 4: 3
        # finally, a single FC layer is used as classifier
        self.untrained_blocks = 2 + 3 + 8 + 36

        # read mean and standard of dataset here to define transforms already
        m, s = readMeanStd(self.mean_std_file)

        # Classification net general and test params
        self.classif_input_size = (3, 227, 227)
        self.classif_test_batch_size = 128
        self.classif_test_pre_proc = True
        self.classif_test_trans = transforms.Compose((transforms.ToTensor(), transforms.Normalize(m, s)))

        # Classification net training params
        self.classif_train_epochs = 0
        self.classif_train_batch_size = 32
        self.classif_train_pre_proc = False
        self.classif_train_aug_rot = r = 45
        self.classif_train_aug_hrange = hr = 0.2
        self.classif_train_aug_vrange = vr = 0.2
        self.classif_train_aug_hsrange = hsr = 0.2
        self.classif_train_aug_vsrange = vsr = 0.2
        self.classif_train_trans = transforms.Compose((random_affine(rotation=r, h_range=hr, v_range=vr, hs_range=hsr, vs_range=vsr), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(m, s)))
        self.classif_lr = 1e-4
        self.classif_momentum = 0.9
        self.classif_weight_decay = 5e-4
        self.classif_optim = 'SGD'
        self.classif_annealing = {30: 0.1}
        self.classif_loss_int = 10
        self.classif_test_int = 100

        # Siamese net general and testing params
        self.siam_input_size = (3, 227, 227)
        self.siam_feature_out_size2d = (8, 8)
        self.siam_feature_dim = 4096
        self.siam_cos_margin = 0  # 0: pi/2 angle, 0.5: pi/3, sqrt(3)/2: pi/6
        self.siam_loss_avg = False
        self.siam_test_batch_size = 64
        self.siam_test_pre_proc = True
        self.siam_test_trans = transforms.Compose((transforms.ToTensor(), transforms.Normalize(m, s)))

        # Siamese net training params
        self.siam_train_trans = transforms.Compose((transforms.ToTensor(), transforms.Normalize(m, s)))
        self.siam_train_pre_proc = True
        self.siam_couples_percentage = 0.9
        self.siam_train_batch_size = 16
        self.siam_lr = 1e-3
        self.siam_momentum = 0.9
        self.siam_weight_decay = 0.0
        self.siam_optim = 'SGD'
        self.siam_annealing = {}
        self.siam_train_epochs = 5
        self.siam_loss_int = 10
        self.siam_test_int = 50

    def save(self, f, prefix):
        f.write('{0}\n'.format(prefix))
        for name, value in sorted(vars(self).items()):
            if name == 'uuid':
                continue
            if name in ('classif_test_trans', 'classif_train_trans', 'siam_test_trans', 'siam_train_trans'):
                f.write('{0}:{1}\n'.format(name, trans_str(value)))
            else:
                f.write('{0}:{1}\n'.format(name, value))
        f.close()

    def save_uuid(self, prefix):
        f = tempfile.NamedTemporaryFile(dir=self.save_dir, delete=False)
        self.save(f, prefix)
        # the following will not work on Windows (would need to add a remove first)
        rename(f.name, path.join(self.save_dir, self.uuid.hex + '.txt'))


test_params = TestParams()


def tensor_t(t, *sizes):
    r = t(*sizes)
    if test_params.cuda_device >= 0:
        return r.cuda()
    else:
        return r.cpu()


def tensor(*sizes):
    return tensor_t(torch.Tensor, *sizes)


# get batches of size batch_size from the set x
def batches(x, batch_size):
    for idx in range(0, len(x), batch_size):
        yield x[idx:min(idx + batch_size, len(x))]


# evaluate a function by batches of size batch_size on the set x
# and fold over the returned values
def fold_batches(f, init, x, batch_size):
    if batch_size <= 0:
        return f(init, 0, x)
    return functools.reduce(lambda last, idx: f(last, idx, x[idx:min(idx + batch_size, len(x))]), range(0, len(x), batch_size), init)


def cos_sim(x1, x2):
    return torch.dot(x1, x2) / (x1.norm() * x2.norm())


# cosine similarity for normed inputs
def cos_sim_normed(x1, x2):
    return torch.dot(x1, x2)


# get couples of images along with their label (same class or not)
# allow given percentage of negative examples as compared to positive
def get_couples(dataset, neg_percentage):
    num_pos = 0
    couples = []
    # get all positive couples
    for (x1, l1), (x2, l2) in itertools.combinations_with_replacement(dataset, 2):
        if l1 == l2:
            num_pos += 1
            couples.append(((x1, x2), 1))
    num_neg = 0
    # get negative couples
    for (x1, l1), (x2, l2) in itertools.combinations(dataset, 2):
        if l1 != l2:
            num_neg += 1
            couples.append(((x1, x2), -1))
            if float(num_neg) / (num_neg + num_pos) >= neg_percentage:
                break
    return couples


# test a classifier model. it should be in eval mode
def test_classif_net(net, testSet, labels, batchSize):
    """
        Test the network accuracy on a testSet
        Return the number of succes and the number of evaluations done
    """
    def eval_batch_test(last, i, batch):
        correct, total = last
        C, H, W = test_params.classif_input_size
        inputs = tensor(len(batch), C, H, W)
        for j, (testIm, _) in enumerate(batch):
            if test_params.classif_test_pre_proc:
                inputs[j] = testIm
            else:
                inputs[j] = test_params.classif_test_trans(testIm)
        outputs = net(Variable(inputs, volatile=True)).data
        _, predicted = torch.max(outputs, 1)
        total += len(batch)
        correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), testSet, batchSize)


def test_print_classif(net, testset_tuple, labels, bestScore=0, epoch=0):
    testTrainSet, testSet = testset_tuple
    net.eval()
    c, t = test_classif_net(net, testSet, labels, test_params.classif_test_batch_size)
    if (c > bestScore):
        bestScore = c
        prefix = 'CLASSIF, EPOCH:{0}, SCORE:{1}'.format(epoch, c)
        test_params.save_uuid(prefix)
        torch.save(net, path.join(test_params.save_dir, test_params.uuid.hex + "_best_classif.ckpt"))
    print("TEST - Correct : ", c, "/", t, '->', float(c) / t)

    c, t = test_classif_net(net, testTrainSet, labels, test_params.classif_test_batch_size)
    torch.save(net, path.join(test_params.save_dir, "model_classif_" + str(epoch) + ".ckpt"))
    print("TRAIN - Correct: ", c, "/", t, '->', float(c) / t)
    net.train()
    return bestScore


def train_classif(net, trainSet, testset_tuple, labels, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train)
            * loss function (criterion)
            * optimizer
    """
    def train_batch(last, i, batch):
        batchCount, score, running_loss = last
        batchSize = len(batch)
        # get the inputs
        C, H, W = test_params.classif_input_size
        inputs = tensor(batchSize, C, H, W)
        for j in range(batchSize):
            if test_params.classif_train_pre_proc:
                inputs[j] = batch[j][0]
            else:
                inputs[j] = test_params.classif_train_trans(batch[j][0])
        inputs = Variable(inputs)

        # get the labels
        lab = tensor_t(torch.LongTensor, len(batchSize))
        for j in range(batchSize):
            lab[j] = labels.index(batch[j][1])
        lab = Variable(lab)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        disp_int = test_params.classif_loss_int
        if batchCount % disp_int == disp_int - 1:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batchCount + 1, running_loss / disp_int))
            running_loss = 0.0

        test_int = test_params.classif_test_int
        if batchCount % test_int == test_int - 1:
            score = test_print_classif(net, testset_tuple, labels, score, epoch + 1)
        return batchCount + 1, score, running_loss

    for epoch in range(test_params.classif_train_epochs):
        # annealing
        if epoch in test_params.classif_annealing:
            default_group = optimizer.state_dict()['param_groups'][0]
            lr = default_group['lr'] * test_params.classif_annealing[epoch]
            momentum = default_group['momentum']
            weight_decay = default_group['weight_decay']
            optimizer = optim.SGD((p for p in net.parameters() if p.requires_grad), lr=lr, momentum=momentum, weight_decay=weight_decay)
        init = 0, bestScore, 0.0
        random.shuffle(trainSet)
        _, bestScore, _ = fold_batches(train_batch, init, trainSet, test_params.classif_train_batch_size)


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (not the average precision/ranking on the ref set). TODO
def test_descriptor_net(net, testSet, testRefSet, is_normalized=True):
    normalize_rows = Normalize2DL2()

    def eval_batch_ref(last, i, batch):
        maxSim, maxLabel, sum_pos, sum_neg, outputs1, testLabels = last
        C, H, W = test_params.siam_input_size
        inputs2 = tensor(len(batch), C, H, W)
        for k, (refIm, _) in enumerate(batch):
            if test_params.siam_test_pre_proc:
                inputs2[k] = refIm
            else:
                inputs2[k] = test_params.siam_test_trans(refIm)
        outputs2 = net(Variable(inputs2, volatile=True)).data
        if not is_normalized:
            outputs2 = normalize_rows(outputs2)
        sim = torch.mm(outputs1, outputs2.t())
        sum_pos += sum(sim[j, k] for j, testLabel in enumerate(testLabels) for k, (_, refLabel) in enumerate(batch) if testLabel == refLabel)
        sum_neg += (sim.sum() - sum_pos)
        batchMaxSim, batchMaxIdx = torch.max(sim, 1)
        for j in range(maxSim.size(0)):
            if (batchMaxSim[j, 0] > maxSim[j, 0]):
                maxSim[j, 0] = batchMaxSim[j, 0]
                maxLabel[j] = batch[batchMaxIdx[j, 0]][1]
        return maxSim, maxLabel, sum_pos, sum_neg, outputs1, testLabels

    def eval_batch_test(last, i, batch):
        correct, total, sum_pos, sum_neg, sum_max, lab_dict = last
        C, H, W = test_params.siam_input_size
        inputs1 = tensor(len(batch), C, H, W)
        for j, (testIm, _) in enumerate(batch):
            if test_params.siam_test_pre_proc:
                inputs1[j] = testIm
            else:
                inputs1[j] = test_params.siam_test_trans(testIm)
        outputs1 = net(Variable(inputs1, volatile=True)).data
        if not is_normalized:
            outputs1 = normalize_rows(outputs1)
        # max similarity, max label, outputs
        maxSim = tensor(len(batch), 1).fill_(-2)
        init = maxSim, [None for _ in batch], sum_pos, sum_neg, outputs1, [lab for im, lab in batch]
        maxSim, maxLabel, sum_pos, sum_neg, _, _ = fold_batches(eval_batch_ref, init, testRefSet, test_params.siam_test_batch_size)
        sum_max += maxSim.sum()
        for j, (_, lab) in enumerate(batch):
            lab_dict[lab].append((maxLabel[j], 1))
        total += len(batch)
        correct += sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(batch))
        return correct, total, sum_pos, sum_neg, sum_max, lab_dict

    lab_dict = dict([(lab, []) for _, lab in testSet])
    return fold_batches(eval_batch_test, (0, 0, 0.0, 0.0, 0.0, lab_dict), testSet, test_params.siam_test_batch_size)


def test_print_siamese(net, testset_tuple, bestScore=0, epoch=0):
    testSet, testRefSet = testset_tuple
    net.eval()
    correct, tot, sum_pos, sum_neg, sum_max, lab_dict = test_descriptor_net(net, testSet, testRefSet)
    # can save labels dictionary (predicted labels for all test labels)
    # for lab in lab_dict:
    #     def red_f(x, y):
    #         return x[0], x[1] + y[1]
    #     L = sorted(lab_dict[lab], key=lambda x: x[0])
    #     g = itertools.groupby(L, key=lambda x: x[0])
    #     red = [reduce(red_f, group) for _, group in g]
    #     lab_dict[lab] = sorted(red, key=lambda x: -x[1])
    # f = open(saveDir + 'lab_dict_' + str(epoch) + '.txt', 'w')
    # for lab in lab_dict:
    #     f.write(str(lab) + ':' + str(lab_dict[lab]) + '\n')
    # f.close()
    num_pos = sum(testLabel == refLabel for _, testLabel in testSet for _, refLabel in testRefSet)
    num_neg = len(testSet) * len(testRefSet) - num_pos
    if (correct > bestScore):
        bestScore = correct
        prefix = 'SIAM, EPOCH:{0}, SCORE:{1}'.format(epoch, correct)
        test_params.save_uuid(prefix)
        torch.save(net, path.join(test_params.save_dir, test_params.uuid.hex + "_best_siam.ckpt"))
    print("TEST - Correct : ", correct, "/", tot, '->', float(correct) / tot, 'avg pos:', sum_pos / num_pos, 'avg neg:', sum_neg / num_neg, 'avg max:', sum_max / len(testSet))

    torch.save(net, path.join(test_params.save_dir, "model_siam_" + str(epoch) + ".ckpt"))

    # training set accuracy
    trainTestSet = testRefSet[:200]
    correct, tot, sum_pos, sum_neg, sum_max, _ = test_descriptor_net(net, trainTestSet, testRefSet)
    num_pos = sum(testLabel == refLabel for _, testLabel in trainTestSet for _, refLabel in testRefSet)
    num_neg = len(trainTestSet) * len(testRefSet) - num_pos
    print("TRAIN - Correct : ", correct, "/", tot, '->', float(correct) / tot, 'avg pos:', sum_pos / num_pos, 'avg neg:', sum_neg / num_neg, 'avg max:', sum_max / len(trainTestSet))
    net.train()
    return bestScore


def train_siamese(net, trainSet, testset_tuple, labels, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    def train_batch(last, i, batch):
        batchCount, score, runningLoss = last

        # using sub-batches (only pairs with biggest loss)
        # losses = []
        # TODO

        # get the inputs
        n = len(batch)
        C, H, W = test_params.siam_input_size
        train_inputs1 = tensor(n, C, H, W)
        train_inputs2 = tensor(n, C, H, W)
        train_labels = tensor(n)
        for j, ((im1, im2), lab) in enumerate(batch):
            if test_params.siam_train_pre_proc:
                train_inputs1[j] = im1
                train_inputs2[j] = im2
            else:
                train_inputs1[j] = test_params.siam_train_trans(im1)
                train_inputs2[j] = test_params.siam_train_trans(im2)
            train_labels[j] = lab

        # zero the parameter gradients, then forward + back prop
        optimizer.zero_grad()
        outputs1, outputs2 = net(Variable(train_inputs1), Variable(train_inputs2))
        loss = criterion(outputs1, outputs2, Variable(train_labels))
        loss.backward()
        optimizer.step()

        # print statistics
        runningLoss += loss.data[0]
        disp_int = test_params.siam_loss_int
        if batchCount % disp_int == disp_int - 1:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batchCount + 1, runningLoss / disp_int))
            runningLoss = 0.0
        # test model every x mini-batches
        test_int = test_params.siam_test_int
        if batchCount % test_int == test_int - 1:
            score = test_print_siamese(net, testset_tuple, score, epoch + 1)
        return batchCount + 1, score, runningLoss

    # loop over the dataset multiple times
    for epoch in range(test_params.siam_train_epochs):
        random.shuffle(trainSet)
        # init = 0, bestScore, 0.0  # batchCount, bestScore, runningLoss
        # _, bestScore, _ = fold_batches(train_batch, init, trainSet, test_params.siam_train_batch_size)
        for i in range(len(trainSet)):
            for j in range(len(trainSet)):
                for k in range(len(trainSet)):
                    if trainSet[i][1] != trainSet[j][1] or trainSet[i][1] == trainSet[k][1]:
                        continue

                    C, H, W = test_params.siam_input_size
                    train_inputs1 = tensor(1, C, H, W)
                    train_inputs2 = tensor(1, C, H, W)
                    train_inputs3 = tensor(1, C, H, W)
                    train_inputs1[0] = trainSet[i][0]
                    train_inputs2[0] = trainSet[j][0]
                    train_inputs3[0] = trainSet[k][0]
                    optimizer.zero_grad()
                    out1, out2, out3 = net(Variable(train_inputs1), Variable(train_inputs2), Variable(train_inputs3))
                    loss = criterion(out1, out2, out3)
                    loss.backward()
                    optimizer.step()

                    print(loss.data[0])


def main():

    def match(x):
        return x.split('/')[-1].split('-')[0]
    # training and test sets (scaled to 300 on the small side)
    trainSetFull = ReadImages.readImageswithPattern(
        test_params.dataset_full, match)
    testSetFull = ReadImages.readImageswithPattern(
        test_params.dataset_full + '/test', match)

    # define the labels list
    listLabel = [t[1] for t in trainSetFull if 'wall' not in t[1]]
    labels = list(set(listLabel))  # we have to give a number for each label

    print('Loading and transforming train/test sets.')
    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    trainSetClassif = []
    for im, lab in trainSetFull:
        im = Image.open(im)
        if test_params.classif_train_pre_proc:
            im = test_params.classif_train_trans(im)
        trainSetClassif.append((im, lab))

    testTrainSetClassif = []
    for im, lab in trainSetFull:
        im = Image.open(im)
        if test_params.classif_test_pre_proc:
            im = test_params.classif_test_trans(im)
        testTrainSetClassif.append((im, lab))

    testSetClassif = []
    for im, lab in testSetFull:
        if lab in labels:
            im = Image.open(im)
            if test_params.classif_test_pre_proc:
                im = test_params.classif_test_trans(im)
            testSetClassif.append((im, lab))

    trainSet = []
    for im, lab in trainSetFull:
        im = Image.open(im)
        if test_params.siam_train_pre_proc:
            im = test_params.siam_train_trans(im)
        trainSet.append((im, lab))

    testSet = []
    for im, lab in testSetFull:
        if lab in labels:
            im = Image.open(im)
            if test_params.siam_test_pre_proc:
                im = test_params.siam_test_trans(im)
            testSet.append((im, lab))

    # transform all training images for testRefSet
    # testRefSet = []
    # for i in range(len(trainSet)):
    #     testRefSet.append((transforms.Scale(227)(trainSet[i][0]), trainSet[i][1]))

    couples = get_couples(trainSet, test_params.siam_couples_percentage)
    num_train = len(couples)
    num_pos = sum(1 for _, lab in couples if lab == 1)
    print('training set size: ', num_train, '#pos:', num_pos, '#neg:', num_train - num_pos)

    if test_params.finetuning:
        class_net = TuneClassif(models.resnet152(pretrained=True), len(labels), untrained_blocks=test_params.untrained_blocks)
    else:
        class_net = models.resnet152()

    class_net = torch.load(path.join(test_params.save_dir, 'best_classif_1.ckpt'))

    if test_params.cuda_device >= 0:
        class_net.cuda()
    else:
        class_net.cpu()
    class_net.train()
    optimizer = optim.SGD((p for p in class_net.parameters() if p.requires_grad), lr=test_params.classif_lr, momentum=test_params.classif_momentum, weight_decay=test_params.classif_weight_decay)
    criterion = nn.loss.CrossEntropyLoss()
    print('Starting classification training')
    testset_tuple = (testTrainSetClassif, testSetClassif)
    # score = test_print_classif(class_net, testset_tuple, labels)
    # TODO try normal weight initialization in classification training (see faster rcnn in pytorch)
    # train_classif(class_net, trainSetClassif, testset_tuple, labels, criterion, optimizer, bestScore=score)
    print('Finished classification training')

    # for ResNet152, spatial feature dimensions are 8x8 (for 227x227 input)
    # for AlexNet, it's 6x6 (for 227x227 input)
    net = Siamese1(class_net, feature_dim=test_params.siam_feature_dim, feature_size2d=test_params.siam_feature_out_size2d)
    if test_params.cuda_device >= 0:
        net.cuda()
    else:
        net.cpu()
    net.train()

    optimizer = optim.SGD((p for p in net.parameters() if p.requires_grad), lr=test_params.siam_lr, momentum=test_params.siam_momentum, weight_decay=test_params.siam_weight_decay)
    # criterion = nn.loss.CosineEmbeddingLoss(margin=test_params.siam_cos_margin, size_average=test_params.siam_loss_avg)
    criterion = TripletLoss(margin=0.2)
    print('Starting descriptor training')
    testset_tuple = (testSet, trainSet)
    score = test_print_siamese(net, testset_tuple, test_params.siam_test_batch_size)
    train_siamese(net, trainSet, testset_tuple, labels, criterion, optimizer, bestScore=score)
    print('Finished descriptor training')


if __name__ == '__main__':
    with torch.cuda.device(test_params.cuda_device):
        main()
