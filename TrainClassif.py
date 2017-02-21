
# coding: utf-8

# In[29]:

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
from ipywidgets import FloatProgress
from IPython.display import display
from __future__ import print_function

from model import ModelDefinition
from dataset import ReadImages, collection
import os
import os.path as path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[30]:

def MeanAndStd(imageList, fname=None):
    """
        Take a list of (image name, label) and return the mean and std dev and save it to a file is fname is set
    """
    imageList = ReadImages.openAll(trainset, (225,225))
    m = collection.ComputeMean(imageList)
    print("Mean : ", m)
    s = collection.ComputeStdDev(imageList, m)
    print("std dev : ", s)
    if not fname is None:
        with open(fname, "w") as f:
            f.write('%.3f %.3f %.3f' %m)
            f.write('\n')
            f.write('%.3f %.3f %.3f' %(s))
    return m,s


# In[31]:

def readMeanStd(fname='data/cli.txt'):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


# In[32]:

def testNet(net, testset, labels, batchSize=32):
    """
        Test the network accuracy on a testset
        Return the number of succes and the number of evaluations done
    """
    net = net.eval() #set the network in eval mode
    correct = 0
    tot = 0
    cpt = 0
    for j in range(len(testset)/batchSize):

        #set the inputs
        inp = torch.Tensor(batchSize,3,225,225).cuda()
        for k in range(batchSize):
            inp[k] = testTransform(testset[j*batchSize+k][0])
            cpt += 1

        #forward pass
        outputs = net(Variable(inp, volatile=True)) #volatile the free memory after the forward

        #compute score
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for k in range(batchSize):
            if (testset[j*batchSize+k][1] in labels):
                correct += (predicted[k][0] == labels.index(testset[j*batchSize+k][1]))
                tot += 1

    #handle the rest of the testset
    rest = len(testset)%batchSize

    #set inputs
    inp = torch.Tensor(rest,3,225,225).cuda()
    for j in range(rest):
        inp[j] = testTransform(testset[len(testset)-rest+j][0])

    #forward
    outputs = mymodel(Variable(inp, volatile=True))

    #compute score
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.tolist()
    for j in range(rest):
        if (testset[len(testset)-rest+j][1] in labels):
           correct += (predicted[j][0] == labels.index(testset[len(testset)-rest+j][1]))
           tot += 1
    return correct, tot


# In[33]:

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
    for epoch in range(epochStart, nbEpoch): # loop over the dataset multiple times
        running_loss = 0.0
        random.shuffle(trainset)    
        for i in range(len(trainset)/batchSize):
            # get the inputs
            inputs = torch.Tensor(batchSize,3,225,225).cuda()
            for j in range(batchSize):
                inputs[j] = imageTransform(trainset[j+i*batchSize][0])
            inputs = Variable(inputs)

            #get the labels
            lab = Variable(torch.LongTensor([labels.index(trainset[j+i*batchSize][1]) for j in range(batchSize)]).cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mymodel(inputs)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 10))
                running_loss = 0.0

            if i % 50 == 49: #test every 20 mini-batches
                print('test :')
                c, t = testNet(mymodel, testset, labels, batchSize=batchSize)
                print("Correct : ", c, "/", t)
                if (c >= bestScore):
                    print("Save best model")
                    best = mymodel
                    bestScore = c
                    torch.save(best, "bestModel.ckpt")
                #else:
                #    mymodel = best
                torch.save(mymodel, path.join(saveDir,"model-"+str(epoch)+".ckpt"))
                mymodel.train() #set the model in train mode

    print('Finished Training')
    return bestScore


# In[34]:

def copyResNet(net, netBase):
    """
        TODO : make more general
    """
    net.conv1.weight.data = netBase.conv1.weight.data
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [(net.layer1, netBase.layer1, 3),
              (net.layer2, netBase.layer2, 4),
              (net.layer3, netBase.layer3, 6),
              (net.layer4, netBase.layer4, 3)
             ]

    for targetLayer, rootLayer, nbC in lLayer:
        for i in range(nbC):
            targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
            targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
            targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
            targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
            targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
            targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            targetLayer[i].conv3.weight.data = rootLayer[i].conv3.weight.data
            targetLayer[i].bn3.weight.data = rootLayer[i].bn3.weight.data
            targetLayer[i].bn3.bias.data = rootLayer[i].bn3.bias.data
        targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
        targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
        targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data 


# In[35]:

def run1():
    
    #training and test sets
    #trainset = ReadImages.readImageswithPattern('/video/CLICIDE', lambda x:x.split('/')[-1].split('-')[0])
    #testset = ReadImages.readImageswithPattern('/video/CLICIDE/test/', lambda x:x.split('/')[-1].split('-')[0])

    trainset = ReadImages.readImageswithPattern('/video/fourviere', lambda x:x.split('/')[-1].split('-')[0])
    testset = ReadImages.readImageswithPattern('/video/fourviere/test/', lambda x:x.split('/')[-1].split('-')[0])
    
    
    #define the labels list
    listLabel = [t[1] for t in trainset if not 'wall' in t[1]] + [t[1] for t in testset if not 'wall' in t[1]]
    labels = list(set(listLabel)) #we have to give a number for each label
    print("There is ", len(labels), " categories")
    
    m, s = readMeanStd('data/fou.txt')
    
    #print("Compute Mean and Std dev on the collection")
    #m, s = MeanAndStd(trainset, "data/fou.txt")
    
    #open the images
    #do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])
        
    for i in range(len(testset)):
        testset[i] = (Image.open(testset[i][0]), testset[i][1])
    
    
    
    
    
    #define the model
    mymodel = models.resnet50(len(labels))
    copyResNet(mymodel, models.resnet50(pretrained=True))
    #ModelDefinition.copyParameters(mymodel, models.alexnet(pretrained=True))
    
    #or load the model
    #mymodel = torch.load('bestModel.ckpt')
    
    criterion = nn.loss.CrossEntropyLoss()
    mymodel.train().cuda()
    
    #define the optimizer to only the classifier with lr of 1e-2
    #optimizer=optim.SGD([
    #                {'params': mymodel.classifier.parameters()},
    #                {'params': mymodel.features.parameters(), 'lr': 0.0}
    #            ], lr=1e-3, momentum=0.9)
    optimizer=optim.SGD([
                    {'params': mymodel.conv1.parameters(),
                     'params': mymodel.bn1.parameters(),
                     'params': mymodel.layer1.parameters(),
                     'params': mymodel.layer2.parameters(),
                     'params': mymodel.layer3.parameters(),
                     'params': mymodel.layer4.parameters()},
                    {'params': mymodel.fc.parameters(), 'lr': 0.01}
                ], lr=0.0, momentum=0.9)

    imageTransform = transforms.Compose( (transforms.Scale(300), transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)) )
    testTransform = transforms.Compose( (transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)))
    batchSize = 64
    
    train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, 
          saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0)


# In[36]:

def run2():
    trainset = ReadImages.readImageswithPattern('/video/fourviere', lambda x:x.split('/')[-1].split('-')[0])
    testset = ReadImages.readImageswithPattern('/video/fourviere/test/', lambda x:x.split('/')[-1].split('-')[0])


    #define the labels list
    listLabel = [t[1] for t in trainset if not 'wall' in t[1]]
    labels = list(set(listLabel)) #we have to give a number for each label
    print("There is ", len(labels), " categories")

    m, s = readMeanStd('data/fou.txt')

    #print("Compute Mean and Std dev on the collection")
    #m, s = MeanAndStd(trainset, "data/fou.txt")

    #open the images
    #do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])

    for i in range(len(testset)):
        testset[i] = (Image.open(testset[i][0]), testset[i][1])

    criterion = nn.loss.CrossEntropyLoss()
    imageTransform = transforms.Compose( (transforms.Scale(300), transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)) )
    testTransform = transforms.Compose( (transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)))
    mymodel = torch.load('bestModel.ckpt')
    mymodel.train().cuda()
    optimizer=optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9)
    train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, 
          saveDir="data/", batchSize=32, epochStart=37, nbEpoch=50, bestScore=308)


# In[45]:

def run3():
    #training and test sets
    trainset = ReadImages.readImageswithPattern('/video/CLICIDE', lambda x:x.split('/')[-1].split('-')[0])
    testset = ReadImages.readImageswithPattern('/video/CLICIDE/test/', lambda x:x.split('/')[-1].split('-')[0])
    
    
    #define the labels list
    listLabel = [t[1] for t in trainset if not 'wall' in t[1]]
    labels = list(set(listLabel)) #we have to give a number for each label
    print("There is ", len(labels), " categories")
    
    m, s = readMeanStd('data/cli.txt')
    
    #print("Compute Mean and Std dev on the collection")
    #m, s = MeanAndStd(trainset, "data/fou.txt")
    
    #open the images
    #do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])
        
    for i in range(len(testset)):
        testset[i] = (Image.open(testset[i][0]), testset[i][1])
    
    
    
    
    
    #define the model
    #mymodel = models.resnet50(len(labels))
    #copyResNet(mymodel, models.resnet50(pretrained=True))
    #ModelDefinition.copyParameters(mymodel, models.alexnet(pretrained=True))
    
    #or load the model
    mymodel = torch.load('bestModel.ckpt')
    
    criterion = nn.loss.CrossEntropyLoss()
    mymodel.train().cuda()
    
    optimizer=optim.SGD([
                    {'params': mymodel.conv1.parameters(),
                     'params': mymodel.bn1.parameters(),
                     'params': mymodel.layer1.parameters(),
                     'params': mymodel.layer2.parameters(),
                     'params': mymodel.layer3.parameters(),
                     'params': mymodel.layer4.parameters()},
                    {'params': mymodel.fc.parameters(), 'lr': 0.01}
                ], lr=0.0, momentum=0.9)

    imageTransform = transforms.Compose( (transforms.Scale(300), transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)) )
    testTransform = transforms.Compose( (transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)))
    batchSize = 64
    
    #best = train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, 
    #      saveDir="data/", batchSize=batchSize, epochStart=0, nbEpoch=40, bestScore=0)
    
    #print("Best score after 40 epoch : ", best)
    
    mymodel = torch.load('bestModel.ckpt')
    mymodel.train().cuda()
    optimizer=optim.SGD([
                    {'params': mymodel.conv1.parameters(),
                     'params': mymodel.bn1.parameters(),
                     'params': mymodel.layer1.parameters(),
                     'params': mymodel.layer2.parameters(),
                     'params': mymodel.layer3.parameters(),
                     'params': mymodel.layer4.parameters()},
                    {'params': mymodel.fc.parameters(), 'lr': 0.001}
                ], lr=0.0, momentum=0.9)

    best = train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, 
          saveDir="data/", batchSize=batchSize, epochStart=40, nbEpoch=50, bestScore=0)
    print("Best score after finetuning : ", best)
    

    #define the optimizer train on all the network
    mymodel = torch.load('bestModel.ckpt')
    mymodel.train().cuda()
    optimizer=optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9)
    best = train(mymodel, trainset, testset, labels, imageTransform, testTransform, criterion, optimizer, 
          saveDir="data/", batchSize=batchSize, epochStart=40, nbEpoch=50, bestScore=best)
    print("Best score after full finetuning : ", best)
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[46]:

run3()


# In[47]:

help(optimizer)

