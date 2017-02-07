
# coding: utf-8

# In[111]:

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


# Process the dataset :
# We have to compute the number of class, and the mean and std for image normalization

# In[112]:

def readImages(imageFile="imList.txt", size=(299,299), openAll=True):
    """
        args :
            imageFile = file with one image path per line
            openAll = bool : load images in memory or not
        ret :
            with openAll : <image path list>, <image list>
            without openAll : <image path list>
    """
    with open(imageFile) as f:
        imList = f.read().splitlines()
        if openAll:
            imOpen = []
            for im in imList:
                i = Image.open(im).resize(size, Image.BILINEAR)
                if openAll:
                    imOpen.append(i)
            return imList, imOpen
        else:
            return imList


# In[119]:

def ComputeMean(imagesList, h=299, w=299):
    """
        TODO : make efficient
    """
    r,g,b = 0,0,0
    toT = transforms.ToTensor()

    #f = FloatProgress(min=0, max=len(imagesList))
    #display(f)

    for im in imagesList:
        #f.value += 1
        t = toT(im)
        for e in t[0].view(-1):
            r += e
        for e in t[1].view(-1):
            g += e
        for e in t[2].view(-1):
            b += e
    return r/(len(imagesList)*h*w), g/(len(imagesList)*h*w), b/(len(imagesList)*h*w) 


# In[123]:

def ComputeStdDev(imagesList, mean):
    """
        TODO : make efficient
    """
    toT = transforms.ToTensor()
    r,g,b = 0,0,0
    h = len(toT(imagesList[0])[0])
    w = len(toT(imagesList[0])[0][0])
    for im in imagesList:
        t = toT(im)
        for e in t[0].view(-1):
            r += (e - mean[0])**2
        for e in t[1].view(-1):
            g += (e - mean[1])**2
        for e in t[2].view(-1):
            b += (e - mean[2])**2
    return (r/(len(imagesList)*h*w))**0.5, (g/(len(imagesList)*h*w))**0.5, (b/(len(imagesList)*h*w))**0.5


# Read the dataset and compute the mean and std dev :

# In[125]:

#trainset, imagesList = readImages("CliList.txt")
#m = ComputeMean(imagesList)
print("Mean : ", m)
#s = ComputeStdDev(imagesList, m)
print("std dev : ", s)


# Define the network as class (from nn.Module) :

# In[137]:

class maxnet(nn.Module):
    def __init__(self, nbClass=464):
        super(maxnet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                nn.ReLU(True),
                nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1)),
                nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                nn.ReLU(True),
                nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1)),
                nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(True),
                nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, nbClass),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#def createMaxnet():
    


# Training

# In[138]:

#create the network
mymodel = maxnet()


# In[139]:

#copy parameters from alexnet
a = models.alexnet(pretrained=True)

#for each feature
for i, f in enumerate(mymodel.features):
    if type(f) is torch.nn.modules.conv.Conv2d:
        #we copy convolution parameters
        f.weight.data = a.features[i].weight.data
        f.bias.data = a.features[i].bias.data

#for each classifier element
for i, f in enumerate(mymodel.classifier):
    if type(f) is torch.nn.modules.linear.Linear:
        #we copy fully connected parameters
        if f.weight.size() == a.classifier[i].weight.size():
            f.weight.data = a.classifier[i].weight.data
            f.bias.data = a.classifier[i].bias.data
        


# In[143]:

#mymodel.cuda()
mymodel = best.train()
criterion = nn.loss.CrossEntropyLoss()
optimizer = optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9)

#trainset, imagesList = readImages("CliList.txt")
#testset, imagesTest = readImages("CliListTest.txt")
#labels = open("CliConcept.txt").read().splitlines()

imageTransform = transforms.Compose( (transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)) )
testTransform = transforms.Compose( (transforms.Scale(225), transforms.ToTensor(), transforms.Normalize(m,s)))
batchSize = 64
bestScore = 0
for epoch in range(50): # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(len(trainset)/batchSize):
        # get the inputs
        elIndex = [random.randrange(0, len(trainset)) for j in range(batchSize)]
        inputs = torch.Tensor(batchSize,3,225,225).cuda()
        for j in range(batchSize):
            inputs[j] = imageTransform(imagesList[elIndex[j]])
        inputs = Variable(inputs)
        lab = Variable(torch.LongTensor([labels.index(trainset[j].split('/')[-1].split('-')[0]) for j in elIndex]).cuda())
        #print(len(lab))
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
            mymodel = mymodel.eval()
            correct = 0
            tot = 0
            cpt = 0
            for j in range(len(testset)/batchSize):
                inp = torch.Tensor(batchSize,3,225,225).cuda()
                for k in range(batchSize):
                    inp[k] = testTransform(imagesTest[j*batchSize+k])
                    cpt += 1
                outputs = mymodel(Variable(inp))
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.tolist()
                for k in range(batchSize):
                    if (testset[j*batchSize+k].split('/')[-1].split('-')[0] in labels):
                        correct += (predicted[k][0] == labels.index(testset[j*batchSize+k].split('/')[-1].split('-')[0]))
                        tot += 1
                        
            rest = len(testset)%batchSize
            inp = torch.Tensor(rest,3,225,225).cuda()
            for j in range(rest):
                inp[j] = testTransform(imagesTest[len(testset)-rest+j])
            outputs = mymodel(Variable(inp))
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            for j in range(rest):
                if (testset[len(testset)-rest+j].split('/')[-1].split('-')[0] in labels):
                   correct += (predicted[j][0] == labels.index(testset[len(testset)-rest+j].split('/')[-1].split('-')[0]))
                   tot += 1
            print("Correct : ", correct, "/", tot)
            if (correct >= bestScore):
                best = mymodel
                bestScore = correct
            #else:
            #    mymodel = best
            mymodel = mymodel.train()
            
print('Finished Training')


# In[424]:

t = torch.Tensor(1,3,225,225).zero_()
mod2 = models.alexnet(pretrained=True)
t[0] = transforms.ToTensor()(imagesList[0])
out1 = mod2(Variable(t))
out2 = mod2(Variable(t))
print("Data : ", out1.data)
print("Max : ", torch.max(out1.data, 1))
print("Out2 : ", out2)
print("Max 2 ", torch.max(out2.data, 1))


# In[495]:

saved = mymodel


# In[477]:

t = torch.Tensor(2,3,224,224)
t[0] = a
t[1] = a


# In[480]:

k = t[0]
j = t[1]
print((k.data == j.data).all())


# In[485]:

mod = models.alexnet(pretrained=True).eval()
t = Variable(t)


# In[486]:

out = mod(t)


# In[487]:

print(out[0])
print(out[1])
print((out.data[0] == out.data[1]).all())


# In[498]:

mymodel.state_dict()


# In[494]:

print(a)


# In[496]:

model_urls = {
    'alexnet': 'https://s3.amazonaws.com/pytorch/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=464):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

