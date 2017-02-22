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

from model.siamese import Siamese1, normalize_rows
from dataset import ReadImages

# TODO create generator to yield couples of images
# / triplets (need a way to identify positive couples for each images,
# then iterate over all others to create triples)
# TODO it seems like naive generator does not work at all
# should train with 'difficult' examples first

# train function then uses these to create batches and trains
# batch by batch on couples/triples

finetuning = False
cos_margin = 0  # angle of 30 degrees (pi/6)


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


def test_feature_net(net, testSet, testRefSet, is_normalized=True, batchSize=1000):
    net.eval()
    def eval_batch_ref(last, i, batch):
        maxSim, maxLabel, outputs1 = last
        inputs2 = torch.Tensor(len(batch), 3, 225, 225).cuda()
        for k, (refIm, _) in enumerate(batch):
            inputs2[k] = refIm
        outputs2 = net(Variable(inputs2, volatile=True)).data
        if not is_normalized:
            normalize_rows(outputs2, False)
        batchMaxSim, batchMaxIdx = torch.max(torch.mm(outputs1, outputs2.t()), 1)
        for j in range(maxSim.size(0)):
            if (batchMaxSim[j, 0] > maxSim[j, 0]):
                maxSim[j, 0] = batchMaxSim[j, 0]
                maxLabel[j] = batch[batchMaxIdx[j, 0]][1]
        return maxSim, maxLabel, outputs1

    def eval_batch_test(last, i, batch):
        correct, total = last
        inputs1 = torch.Tensor(len(batch), 3, 225, 225).cuda()
        for j, (testIm, _) in enumerate(batch):
            inputs1[j] = testIm
        outputs1 = net(Variable(inputs1, volatile=True)).data
        if not is_normalized:
            normalize_rows(outputs1, False)
        # max similarity, max label, outputs
        maxSim = torch.Tensor(len(batch), 1).cuda()
        maxSim.fill_(-2.0)
        init = maxSim, [None for _ in batch], outputs1
        maxSim, maxLabel, _ = fold_batches(eval_batch_ref, init, testRefSet, batchSize)
        total += len(batch)
        correct += sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), testSet, batchSize)


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
    random.shuffle(couples)
    return couples


def readMeanStd(fname='data/cli.txt'):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std



def train(mymodel, trainset, testset_tuple, labels, trainTransform, testTransform, criterion, optimizer, saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0):
    """
        Train a network
        inputs :
            * trainset
            * testset,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    for epoch in range(epochStart, nbEpoch): # loop over the dataset multiple times
        random.shuffle(trainset)

        def train_batch(last, i, batch):
            batchCount, bestScore, runningLoss = last
            batchSize = len(batch)
            # test model every x mini-batches
            if batchCount % 50 == 0:
                correct, tot = test_feature_net(mymodel, testset_tuple[0], testset_tuple[1])
                print("Correct : ", correct, "/", tot, '->', float(correct) / tot)
                if (correct > bestScore):
                    bestModel = mymodel
                    bestScore = correct
                    torch.save(bestModel, "bestModel.ckpt")
                #else:
                #    mymodel = best
                torch.save(mymodel, path.join(saveDir,"model-"+str(epoch)+".ckpt"))
                mymodel.train() #set the model in train mode

            # using sub-batches (only pairs with biggest loss)
            # losses = []
            # inputs1 = torch.Tensor(batchSize,3,225,225).cuda()
            # inputs2 = torch.Tensor(batchSize,3,225,225).cuda()
            # labels = torch.LongTensor(batchSize, 1).cuda()
            # for j in range(batchSize):
            #     inputs1[j] = trainTransform(trainset[i+j][0][0])
            #     inputs2[j] = trainTransform(trainset[i+j][0][1])
            #     labels[j] = trainset[i+j][1]
            #     #get the label
            #     lab = Variable(labels[j])
            #     input1 = inputs1[j].view(*([1L] + list(inputs1[j].size())))
            #     input2 = inputs2[j].view(*([1L] + list(inputs1[j].size())))
            #     outputs = mymodel(Variable(input1), Variable(input2))
            #     loss = criterion(outputs, lab)
            #     losses.append((loss, j))
            # losses.sort(key=lambda x: x[0].data[0], reverse=True)

            # get the inputs
            # TODO image transforms take a long time, see if this can be made faster (>1s vs 0.1s for the rest of train)
            train_inputs1 = torch.Tensor(batchSize,3,225,225).cuda()
            train_inputs2 = torch.Tensor(batchSize,3,225,225).cuda()
            for j in range(batchSize):
                train_inputs1[j] = trainTransform(batch[j][0][0])
                train_inputs2[j] = trainTransform(batch[j][0][1])

            train_labels = torch.Tensor(batchSize).cuda()
            for j in range(batchSize):
                train_labels[j] = batch[j][1]

            # zero the parameter gradients, then forward + back prop
            optimizer.zero_grad()
            outputs1, outputs2 = mymodel(Variable(train_inputs1), Variable(train_inputs2))
            loss = criterion(outputs1, outputs2, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # print statistics
            runningLoss += loss.data[0]
            if batchCount % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, batchCount+1, runningLoss / 10))
                runningLoss = 0.0
            return batchCount + 1, bestScore, runningLoss

        init = 0, bestScore, 0.0  # batchCount, bestScore, runningLoss
        fold_batches(train_batch, init, trainset, batchSize)

    print('Finished Training')


if __name__ == '__main__':

    # training and test sets
    trainset = ReadImages.readImageswithPattern(
        '/video/CLICIDE', lambda x: x.split('/')[-1].split('-')[0])
    testSetFull = ReadImages.readImageswithPattern(
        '/video/CLICIDE/test/', lambda x: x.split('/')[-1].split('-')[0])

    m, s = readMeanStd('data/cli.txt')

    # define the labels list
    listLabel = [t[1] for t in trainset if 'wall' not in t[1]]
    labels = list(set(listLabel))  # we have to give a number for each label

    testTransform = transforms.Compose((transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m, s)))
    trainTransform = transforms.Compose((transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m, s)))

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

    # transform all training images for testset
    testRefSet = []
    for i in range(len(trainset)):
        testRefSet.append((testTransform(trainset[i][0]), trainset[i][1]))

    # apply upscaling transform to train set already
    # as this takes a long time
    for i in range(len(trainset)):
        trainset[i] = (transforms.Scale(300)(trainset[i][0]), trainset[i][1])

    # define the model
    # mymodel = ModelDefinition.Maxnet()
    # ModelDefinition.copyParameters(mymodel, models.alexnet(pretrained=True))

    couples = get_couples(trainset)
    num_train = len(couples)
    num_pos = sum(1 for _, lab in couples if lab == 1)
    print('training set size: ', num_train, '(pos:', num_pos, 'neg:', num_train-num_pos, ')')
    criterion = nn.loss.CosineEmbeddingLoss(margin=cos_margin)

    if finetuning:
        # or load the model
        mymodel = Siamese1(models.alexnet(pretrained=True))
        mymodel.train().cuda()

        # define the optimizer to only the classifier with lr of 1e-2
        params = [
            {'params': mymodel.classifier.parameters()},
            {'params': mymodel.features.parameters(), 'lr': 0.0}
            ]
        if mymodel.final_features:
            params = [
                {'params': mymodel.final_features.parameters()},
                {'params': mymodel.classifier.parameters()},
                {'params': mymodel.features.parameters(), 'lr': 0.0}
                ]
        optimizer = optim.SGD(params, lr=1e-2, momentum=0.9)

        batchSize = 1024
        train(mymodel, couples, (testSet, testRefSet), labels, trainTransform, testTransform, criterion, optimizer, batchSize=batchSize)
        # define the optimizer train on all the network
        optimizer = optim.SGD(mymodel.parameters(), lr=1e-4, momentum=0.9)
        train(mymodel, couples, (testSet, testRefSet), labels, trainTransform, testTransform, criterion, optimizer, batchSize=batchSize)
    else:
        mymodel = Siamese1(models.alexnet())
        mymodel.train().cuda()

        # define the optimizer to learn everything
        optimizer = optim.SGD(mymodel.parameters(), lr=1e-2, momentum=0.9)

        batchSize = 1024
        # use the couples as train and test set
        train(mymodel, couples, (testSet, testRefSet), labels, trainTransform, testTransform, criterion, optimizer, batchSize=batchSize)
