
# coding: utf-8

import torch.nn as nn
import torchvision.models as models


def normalize_rows(x, is_variable=True):
    if is_variable:
        for row in x:
            row.div_(row.norm().data[0])
    else:
        for row in x:
            row.div_(row.norm())


class Siamese1(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier
    """
    def __init__(self, net, feature_dim=100):
        super(Siamese1, self).__init__()
        self.features = net.features
        self.classifier = net.classifier
        if feature_dim <= 0:
            self.final_features = None
        else:
            self.final_features = nn.Linear(net.classifier[len(net.classifier._modules)-1].out_features, feature_dim)

    def forward_single(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        normalize_rows(x)
        if self.final_features:
            x = self.final_features(x)
            normalize_rows(x)
        return x

    def forward(self, x1, x2=None):
        if self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)

# In[39]:

def siamese1():
    return Siamese1(models.alexnet(pretrained=True))
