
# coding: utf-8

# In[43]:

import torch
import torch.nn as nn
import torchvision.models as models


# In[38]:

class Siamese1(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    def __init__(self, net):
        super(siamese, self).__init__()
        self.features = net
        self.classifier = nn.Linear(net.classifier[len(net.classifier._modules)-1].out_features*2, 1)
    
    def forward(self, x1, x2):
        x = torch.cat( (self.features(x1), self.features(x2)), 1)
        x = self.classifier(x)
        return x


# In[49]:

class Siamese2(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    def __init__(self, net):
        super(Siamese2, self).__init__()
        self.features = net
        
    def forward(self, x1, x2):
        return (self.features(x1), self.features(x2))


# In[39]:

def siamese1():
    return Siamese1(models.alexnet(pretrained=True))


# In[50]:

t = Variatorch.Tensor(1,3,225,225)
s = Siamese2(models.alexnet(pretrained=True))
o = s(t, t)


# In[ ]:

print(o)

