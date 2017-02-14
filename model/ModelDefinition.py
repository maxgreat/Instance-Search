
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.parallel


# In[2]:

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
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nbClass),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# In[ ]:

def Maxnet(nbClass=464):
    return maxnet(nbClass)


# In[ ]:

def copyParameters(net, modelBase):
    """
        Copy parameters from a model to another
    """
    #for each feature
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            #we copy convolution parameters
            f.weight.data = modelBase.features[i].weight.data
            f.bias.data = modelBase.features[i].bias.data

    #for each classifier element
    for i, f in enumerate(net.classifier):
        if type(f) is torch.nn.modules.linear.Linear:
            #we copy fully connected parameters
            if f.weight.size() == modelBase.classifier[i].weight.size():
                f.weight.data = modelBase.classifier[i].weight.data
                f.bias.data = modelBase.classifier[i].bias.data

