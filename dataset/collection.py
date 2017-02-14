
# coding: utf-8

# In[1]:

import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms


# In[ ]:

def ComputeMean(imagesList, h=299, w=299):
    """
        TODO : make efficient
        Return the mean of the collection for each chanel RGB
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


# In[ ]:

def ComputeStdDev(imagesList, mean):
    """
        TODO : make efficient
        Return the std deviation for each channel over the collection
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


# In[2]:

def createConceptDict(imageList):
    """
        Create a dictionnary that store for each concept the list of image path corresponding
    """
    ConceptDict = {}
    for im in imageList:
        if im[1] in ConceptDict.keys():
            ConceptDict[im[1]].append(im[0])
        else:
            ConceptDict[im[1]] = [im[0]]
    return ConceptDict

