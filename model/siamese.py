
# coding: utf-8

# In[8]:

import torch
import torch.nn as nn
import torchvision.models as models
import math

cosine_dist = True
cos_margin = math.sqrt(3) / 2  # angle of 30 degrees (pi/6)

# In[38]:

class Siamese1(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier
    """
    def __init__(self, net):
        super(Siamese1, self).__init__()
        self.features = net
        n_out = net.classifier[len(net.classifier._modules)-1].out_features*2
        self.classifier = nn.Linear(n_out, 2)

    def forward(self, x1, x2):
        if cosine_dist:
            return self.features(x1), self.features(x2)
        else:
            x = torch.cat((self.features(x1), self.features(x2)), 1)
            x = self.classifier(x)
            return x


# In[39]:

def siamese1():
    return Siamese1(models.alexnet(pretrained=True))
