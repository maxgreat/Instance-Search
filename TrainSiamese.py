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
from os import path

from PIL import Image

from model.siamese import Siamese1, TuneClassif, Normalize2DL2, MetricLoss
from dataset import ReadImages

# TODO create generator to yield couples of images
# / triplets (need a way to identify positive couples for each images,
# then iterate over all others to create triples)
# TODO it seems like naive generator does not work at all
# should train with 'difficult' examples first

used_dataset = 'fourviere'
finetuning = True
cos_margin = math.sqrt(0) / 2  # 0: pi/2 angle, 0.5: pi/3, sqrt(3)/2: pi/6


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
def get_couples(dataset, neg_percentage=0.9):
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


def readMeanStd(fname='data/cli.txt'):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


# test a classifier model. it should be in eval mode
def test_classif_net(net, testSet, labels, batchSize):
    """
        Test the network accuracy on a testSet
        Return the number of succes and the number of evaluations done
    """
    def eval_batch_test(last, i, batch):
        correct, total = last
        inputs = torch.Tensor(len(batch), 3, 227, 227).cuda()
        for j, (testIm, _) in enumerate(batch):
            inputs[j] = testIm
        outputs = net(Variable(inputs, volatile=True)).data
        _, predicted = torch.max(outputs, 1)
        total += len(batch)
        correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), testSet, batchSize)


def test_print_classif(net, testSet, labels, bestScore=0, saveDir='data/', epoch=0, batchSize=1000):
    net.eval()
    c, t = test_classif_net(net, testSet, labels, batchSize=batchSize)
    print("Correct : ", c, "/", t, '->', float(c) / t)
    if (c >= bestScore):
        best = net
        bestScore = c
        torch.save(best, "bestModel.ckpt")
    # else:
    #    net = best
    torch.save(net, path.join(saveDir, "model-" + str(epoch) + ".ckpt"))
    net.train()
    return bestScore


def train_classif(net, trainset, testSet, labels, trainTransform, criterion, optimizer, saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0):
    """
        Train a network
        inputs :
            * trainset
            * testSet,
            * transformations to apply to image (for train)
            * loss function (criterion)
            * optimizer
    """
    def train_batch(last, i, batch):
        batchCount, bestScore, running_loss = last
        batchSize = len(batch)
        # get the inputs
        inputs = torch.Tensor(batchSize, 3, 227, 227).cuda()
        for j in range(batchSize):
            inputs[j] = trainTransform(batch[j][0])
        inputs = Variable(inputs)

        # get the labels
        lab = Variable(torch.LongTensor([labels.index(batch[j][1]) for j in range(batchSize)]).cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if batchCount % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batchCount + 1, running_loss / 10))
            running_loss = 0.0

        if batchCount % 50 == 49:  # test every x mini-batches
            bestScore = test_print_classif(net, testSet, labels, bestScore, saveDir, epoch)
        return batchCount + 1, bestScore, running_loss

    for epoch in range(epochStart, nbEpoch):  # loop over the dataset multiple times
        init = 0, bestScore, 0.0
        random.shuffle(trainset)
        fold_batches(train_batch, init, trainset, batchSize)
    print('Finished classification training')


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (not the average precision/ranking on the ref set). TODO
def test_descriptor_net(net, testSet, testRefSet, batchSize, is_normalized=True):
    normalize_rows = Normalize2DL2()

    def eval_batch_ref(last, i, batch):
        maxSim, maxLabel, sum_pos, sum_neg, outputs1, testLabels = last
        inputs2 = torch.Tensor(len(batch), 3, 227, 227).cuda()
        for k, (refIm, _) in enumerate(batch):
            inputs2[k] = refIm
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
        correct, total, sum_pos, sum_neg = last
        inputs1 = torch.Tensor(len(batch), 3, 227, 227).cuda()
        for j, (testIm, _) in enumerate(batch):
            inputs1[j] = testIm
        outputs1 = net(Variable(inputs1, volatile=True)).data
        if not is_normalized:
            outputs1 = normalize_rows(outputs1)
        # max similarity, max label, outputs
        maxSim = torch.Tensor(len(batch), 1).cuda()
        maxSim.fill_(-2.0)
        init = maxSim, [None for _ in batch], sum_pos, sum_neg, outputs1, [x[1] for x in batch]
        maxSim, maxLabel, sum_pos, sum_neg, _, _ = fold_batches(eval_batch_ref, init, testRefSet, batchSize)
        total += len(batch)
        correct += sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(batch))
        return correct, total, sum_pos, sum_neg

    return fold_batches(eval_batch_test, (0, 0, 0.0, 0.0), testSet, batchSize)


def test_siamese_print(net, testset_tuple, batchSize, bestScore=0, saveDir='data/', epoch=0):
    testSet, testRefSet = testset_tuple
    net.eval()
    correct, tot, sum_pos, sum_neg = test_descriptor_net(net, testSet, testRefSet, batchSize)
    num_pos = sum(testLabel == refLabel for _, testLabel in testSet for _, refLabel in testRefSet)
    num_neg = len(testSet) * len(testRefSet) - num_pos
    print("TEST SET - Correct : ", correct, "/", tot, '->', float(correct) / tot, 'avg pos:', sum_pos / num_pos, 'avg neg:', sum_neg / num_neg)
    if (correct > bestScore):
        bestModel = net
        bestScore = correct
        torch.save(bestModel, "bestModel.ckpt")
    # else:
    #    net = best
    torch.save(net, path.join(saveDir, "model-" + str(epoch) + ".ckpt"))

    # training set accuracy
    trainTestSet = testRefSet[:200]
    correct, tot, sum_pos, sum_neg = test_descriptor_net(net, trainTestSet, testRefSet, batchSize)
    num_pos = sum(testLabel == refLabel for _, testLabel in trainTestSet for _, refLabel in testRefSet)
    num_neg = len(trainTestSet) * len(testRefSet) - num_pos
    print("TRAIN SET - Correct : ", correct, "/", tot, '->', float(correct) / tot, 'avg pos:', sum_pos / num_pos, 'avg neg:', sum_neg / num_neg)
    net.train()
    return bestScore


def train_siamese(net, trainset, testset_tuple, labels, trainTransform, criterion, optimizer, saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0):
    """
        Train a network
        inputs :
            * trainset
            * testSet,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    def train_batch(last, i, batch):
        batchCount, bestScore, runningLoss = last

        # using sub-batches (only pairs with biggest loss)
        # losses = []
        # TODO

        # get the inputs
        n = len(batch)
        train_inputs1 = torch.Tensor(n, 3, 227, 227).cuda()
        train_inputs2 = torch.Tensor(n, 3, 227, 227).cuda()
        train_labels = torch.Tensor(n).cuda()
        for j, ((im1, im2), lab) in enumerate(batch):
            train_inputs1[j] = trainTransform(im1)
            train_inputs2[j] = trainTransform(im2)
            train_labels[j] = lab

        # zero the parameter gradients, then forward + back prop
        optimizer.zero_grad()
        outputs1, outputs2 = net(Variable(train_inputs1), Variable(train_inputs2))
        loss = criterion(outputs1, outputs2, Variable(train_labels))
        loss.backward()
        optimizer.step()

        # print statistics
        runningLoss += loss.data[0]
        if batchCount % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batchCount + 1, runningLoss / 10))
            runningLoss = 0.0
        # test model every x mini-batches
        if batchCount % 50 == 49:
            bestScore = test_siamese_print(net, testset_tuple, batchSize, bestScore, saveDir, epoch)
        return batchCount + 1, bestScore, runningLoss

    # loop over the dataset multiple times
    for epoch in range(epochStart, nbEpoch):
        random.shuffle(trainset)
        init = 0, bestScore, 0.0  # batchCount, bestScore, runningLoss
        fold_batches(train_batch, init, trainset, batchSize)

    print('Finished descriptor training')


if __name__ == '__main__':

    # training and test sets
    def match(x):
        return x.split('/')[-1].split('-')[0]
    trainset = ReadImages.readImageswithPattern(
        'data/pre_proc/' + used_dataset, match)
    testSetFull = ReadImages.readImageswithPattern(
        '/video/' + used_dataset + '/test/', match)

    m, s = readMeanStd('data/' + ('cli.txt' if used_dataset == 'CLICIDE' else 'fou.txt'))

    # define the labels list
    listLabel = [t[1] for t in trainset if 'wall' not in t[1]]
    labels = list(set(listLabel))  # we have to give a number for each label

    testTransform = transforms.Compose((transforms.Scale(227), transforms.CenterCrop(227), transforms.ToTensor(), transforms.Normalize(m, s)))
    # upscaling was already applied in train set
    trainTransform = transforms.Compose((transforms.RandomCrop(227), transforms.ToTensor(), transforms.Normalize(m, s)))

    print('Loading and transforming train/test sets. This can take a while.')

    # open the images
    # do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])

    # transform the test images already
    testSet = []
    for i in range(len(testSetFull)):
        if testSetFull[i][1] in labels:
            testSet.append((testTransform(Image.open(testSetFull[i][0])), testSetFull[i][1]))

    # transform all training images for testSet
    testRefSet = []
    for i in range(len(trainset)):
        testRefSet.append((testTransform(trainset[i][0]), trainset[i][1]))

    couples = get_couples(trainset)
    num_train = len(couples)
    num_pos = sum(1 for _, lab in couples if lab == 1)
    print('training set size: ', num_train, '(pos:', num_pos, 'neg:', num_train - num_pos, ')')

    if finetuning:
        class_net = TuneClassif(models.alexnet(pretrained=True), len(labels))
        lr = 1e-3
    else:
        class_net = models.alexnet()
        lr = 1e-2
    class_net.train().cuda()
    optimizer = optim.SGD((p for p in class_net.parameters() if p.requires_grad), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.loss.CrossEntropyLoss()
    test_print_classif(class_net, testSet, labels)
    # TODO try normal weight initialization in classification training (see faster rcnn in pytorch)
    train_classif(class_net, trainset, testSet, labels, trainTransform, criterion, optimizer, nbEpoch=5)

    net = Siamese1(class_net, feature_dim=0)
    net.train().cuda()

    optimizer = optim.SGD((p for p in net.parameters() if p.requires_grad), lr=1e-3, momentum=0.9)
    criterion = nn.loss.CosineEmbeddingLoss(margin=cos_margin)
    batchSize = 256
    test_siamese_print(net, (testSet, testRefSet), batchSize)
    train_siamese(net, couples, (testSet, testRefSet), labels, trainTransform, criterion, optimizer, batchSize=batchSize, nbEpoch=1)
