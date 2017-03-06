
# coding: utf-8

# In[1]:

from __future__ import division
import glob
import os.path as path
from PIL import Image
import torchvision.transforms as transforms


# In[2]:

def readImagesInCLass(folder='.'):
    """
        Read a folder containing images with the structure :
            folder
                --class1
                    --image1
                    --image2
                --class2
                    --image3
                    --image3
        
        Return :
            list of couple : (image, class)
    """
    
    exts = ('*.jpg', '*.JPG', '*.JPEG', "*.png")
    r = []
    for el in glob.iglob(path.join(folder, '*')):
        if path.isdir(el):
            for ext in exts:
                r.extend( [(im, el.split('/')[-1]) for im in glob.iglob(path.join(el, ext)) ] )
    return r


# In[3]:

def readImageswithPattern(folder='.', matchFunc=lambda x:x.split('.')[0]):
    """
        Read a folder containing images where the name of the class is in the filename
        the match function should return the class given the filename
        Return :
            list of couple : (image, class)
    """
    exts = ('*.jpg', '*.JPG', '*.JPEG', "*.png")
    r = []
    for ext in exts:
        r.extend( [(im, matchFunc(im)) for im in glob.iglob(path.join(folder, ext)) ] )
    return r


# In[4]:

def openAll(imageList, size=0 ):
    """
        Open all images, return a list of PIL images
    """
    if size == 0:
        return [Image.open(im) for im, c in imageList]
    else:
        return [Image.open(im).resize(size) for im, c in imageList]
        


# In[ ]:

def openDict(imageList, size=(225,225)):
    """
        Open all images, return a dictionnary of (image name : PIL image) and resize as the given size
    """
    return {im: Image.open(im).resize(size) for im, c in imageList}


# In[5]:

def positiveCouples(dataset):
    """
        Create all positive couples in the dataset
    """
    return [ (im[0], im2[0], 1) for im in dataset for im2 in dataset if im[1]==im2[1]]


# In[6]:

def negativeCouples(dataset):
    """
        Create all negative couples in the dataset
    """
    return [ (im[0], im2[0], -1) for im in dataset for im2 in dataset if im[1] != im2[1]]


# In[9]:

def createCouples(dataset):
    """
        Create all couples in the dataset
    """
    return [ (im[0], im2[0], 1) if im[1] == im2[1] else (im[0], im2[0], -1) for im in dataset for im2 in dataset]


# In[10]:

if __name__ == '__main__':
    dataset = readImageswithPattern('/video/CLICIDE', lambda x:x.split('/')[-1].split('-')[0]) #read Clicide dataset
    p = positiveCouples(dataset) #Clicide positives couples
    print(len(p)) #should be 27217
    n = negativeCouples(dataset) #Clicide negatives couples, all of them
    print(len(n)) #should be 10502754 (10M)
    print("Nb of p / nb of n : %.3f %%"  % (len(p)/len(n)*100)) #around 0.2% of positive examples
    a = createCouples(dataset)
    print(len(a))
    

