import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

import itertools
import random
import time
from os import path

from PIL import Image

from model.siamese import Siamese1, cosine_dist, cos_margin
from dataset import ReadImages

# TODO create generator to yield couples of images
# / triplets (need a way to identify positive couples for each images,
# then iterate over all others to create triples)
# TODO it seems like naive generator does not work at all
# should train with 'difficult' examples first

# train function then uses these to create batches and trains
# batch by batch on couples/triples

finetuning = False


# get batches of size batch_size from the set x
def batches(x, batch_size):
    for idx in range(0, len(x), batch_size):
        yield x[idx:min(idx + batch_size, len(x))]


# evaluate a function by batches of size batch_size on the set x
def eval_by_batches(f, x, batch_size):
    if (batch_size <= 0):
        f(0, x)
        return
    for idx in range(0, len(x), batch_size):
        f(idx, x[idx:min(idx + batch_size, len(x))])


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
            couples.append(((x1, x2), -1 if cosine_dist else 0))
            if float(num_neg) / (num_neg + num_pos) >= neg_percentage:
                break
    random.shuffle(couples)
    return couples


def readMeanStd(fname='data/cli.txt'):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


def testNet(net, testset, labels, batchSize=32):
    """
        Test the network accuracy on a testset
        Return the number of succes and the number of evaluations done
    """
    net = net.eval() #set the network in eval mode
    correct = [0]  # avoid scope issues
    tot = [0]
    pos_values, neg_values = [], []

    values = torch.Tensor(len(testset)).cuda()

    def get_values(i, batch):
        batchSize = len(batch)
        inputs1 = torch.Tensor(batchSize,3,225,225).cuda()
        inputs2 = torch.Tensor(batchSize,3,225,225).cuda()
        for k in range(batchSize):
            inputs1[k] = testTransform(testset[i+k][0][0])
            inputs2[k] = testTransform(testset[i+k][0][1])
        get_el = None
        comp = None
        if cosine_dist:
            outputs1, outputs2 = net(Variable(inputs1, volatile=True), Variable(inputs2, volatile=True))
            for k in range(batchSize):
                o1, o2 = outputs1.data[k], outputs2.data[k]
                values[k] = torch.dot(o1, o2) / (o1.norm() * o2.norm())

            def get_el(x, i):
                return x[i]

            def comp(x1, x2):
                return x1 * x2 > 0
        else:
            outputs = net(Variable(inputs1, volatile=True), Variable(inputs2, volatile=True))
            values, predicted = torch.max(outputs.data, 1)

            def get_el(x, i):
                return x[i,0]

            def comp(x1, x2):
                return x1 == x2
        for k in range(batchSize):
            # if testset[i+k][1] == 1 and random.random() < 0.05:
            #     print('pos:', get_el(predicted, k), get_el(values, k))
            # if testset[i+k][1] != 1 and random.random() < 0.005:
            #     print('neg:', get_el(predicted, k), get_el(values, k))

            if testset[i+k][1] == 1:
                pos_values.append(get_el(values, k))
            else:
                neg_values.append(get_el(values, k))
            correct[0] += comp(get_el(predicted, k), testset[i+k][1])
            tot[0] += 1

    eval_by_batches(get_values, testset, batchSize)
    print('mean pos: ', sum(pos_values) / float(len(pos_values)), 'mean neg: ', sum(neg_values) / float(len(neg_values)))
    return correct[0], tot[0]


def train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0):
    """
        Train a network
        inputs :
            * trainset
            * testset,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    best = [bestScore]
    for epoch in range(epochStart, nbEpoch): # loop over the dataset multiple times
        running_loss = [0.0]
        batchCount = [0]
        random.shuffle(trainset)

        def train_batch(i, batch):
            batchSize = len(batch)

            # test model every x mini-batches
            if batchCount[0] % 100 == 0:
                print('test :')
                correct, tot = testNet(mymodel, testset, labels, batchSize=batchSize)
                print("Correct : ", correct, "/", tot)
                if (correct >= best[0]):
                    bestModel = mymodel
                    best[0] = correct
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
            #     inputs1[j] = imageTransform(trainset[i+j][0][0])
            #     inputs2[j] = imageTransform(trainset[i+j][0][1])
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
                train_inputs1[j] = imageTransform(trainset[i+j][0][0])
                train_inputs2[j] = imageTransform(trainset[i+j][0][1])

            if cosine_dist:
                train_labels = torch.Tensor(batchSize).cuda()
                for j in range(batchSize):
                    train_labels[j] = trainset[i+j][1]
            else:
                train_labels = torch.LongTensor(batchSize).cuda()
                for j in range(batchSize):
                    train_labels[j] = trainset[i+j][1]

            # zero the parameter gradients, then forward + back prop
            optimizer.zero_grad()
            if cosine_dist:
                outputs1, outputs2 = mymodel(Variable(train_inputs1), Variable(train_inputs2))
                loss = criterion(outputs1, outputs2, Variable(train_labels))
            else:
                outputs = mymodel(Variable(train_inputs1), Variable(train_inputs2))
                loss = criterion(outputs, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss[0] += loss.data[0]
            if batchCount[0] % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, batchCount[0]+1, running_loss[0] / 10))
                running_loss[0] = 0.0

            batchCount[0] += 1
        eval_by_batches(train_batch, trainset, batchSize)

    print('Finished Training')


if __name__ == '__main__':

    # training and test sets
    trainset = ReadImages.readImageswithPattern(
        '/video/CLICIDE', lambda x: x.split('/')[-1].split('-')[0])
    testset = ReadImages.readImageswithPattern(
        '/video/CLICIDE/test/', lambda x: x.split('/')[-1].split('-')[0])

    m, s = readMeanStd('data/cli.txt')

    # define the labels list
    listLabel = [t[1] for t in trainset if 'wall' not in t[1]]
    labels = list(set(listLabel))  # we have to give a number for each label

    # open the images
    # do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])

    for i in range(len(testset)):
        testset[i] = (Image.open(testset[i][0]), testset[i][1])

    # define the model
    # mymodel = ModelDefinition.Maxnet()
    # ModelDefinition.copyParameters(mymodel, models.alexnet(pretrained=True))

    couples = get_couples(trainset)
    print('training set size: ', len(couples))
    couples_test = get_couples(testset, 1.0)
    print('test set size: ', len(couples_test))
    imageTransform = transforms.Compose((transforms.Scale(300), transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m, s)))
    testTransform = transforms.Compose((transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m, s)))
    if cosine_dist:
        criterion = nn.loss.CosineEmbeddingLoss(margin=cos_margin)
    else:
        criterion = nn.loss.CrossEntropyLoss()
    if finetuning:
        # or load the model
        mymodel = Siamese1(models.alexnet(pretrained=True))

        mymodel.train().cuda()

        # define the optimizer to only the classifier with lr of 1e-2
        optimizer = optim.SGD([
            {'params': mymodel.classifier.parameters()},
            {'params': mymodel.features.parameters(), 'lr': 0.0}
        ], lr=1e-2, momentum=0.9)

        batchSize = 64
        train(mymodel, couples, couples_test, labels, imageTransform, testTransform, criterion, optimizer, batchSize=batchSize)
        # define the optimizer train on all the network
        optimizer = optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9)
        train(mymodel, couples, couples_test, labels, imageTransform, testTransform, criterion, optimizer, batchSize=batchSize)
    else:
        mymodel = Siamese1(models.alexnet(pretrained=True))
        mymodel.train().cuda()

        # define the optimizer to learn everything
        optimizer = optim.SGD(mymodel.parameters(), lr=1e-4, momentum=0.9)

        # use the couples as train and test set
        train(mymodel, couples, couples_test, labels, imageTransform, testTransform, criterion, optimizer)
