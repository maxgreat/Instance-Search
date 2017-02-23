
# coding: utf-8

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import numpy as np


class NormalizeRows(Function):

    def forward(self, input):
        self.buf = input.clone().norm(2, 1).expand_as(input)
        return input / self.buf

    def backward(self, grad_output):
        return grad_output / self.buf


class NormalizeRowsModule(nn.Module):

    def __init__(self):
        super(NormalizeRowsModule, self).__init__()

    def forward(self, input):
        return NormalizeRows()(input)


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
            self.final_features = NormalizeRowsModule()
        else:
            self.final_features = nn.Sequential(
                NormalizeRowsModule(),
                nn.Linear(net.classifier[len(net.classifier._modules)-1].out_features, feature_dim),
                NormalizeRowsModule()
            )

    def forward_single(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.final_features(x)
        return x

    def forward(self, x1, x2=None):
        if self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)

# In[39]:

def siamese1():
    return Siamese1(models.alexnet(pretrained=True))


# metric loss according to Chopra et al "Learning a Similarity Metric Discriminatively, with Application to Face Verification"
# since we assume normalized vectors, we use Q=2
class MetricL(Function):

    def __init__(self, size_average=True):
        super(MetricL, self).__init__()
        self.size_average = size_average

    # TODO: everything could be done inplace,
    # more difficult though (for norm see torch Cosine Loss)
    def terms(self, input1, input2, y):
        diff = input1 - input2
        energy = diff.norm(1, 1)
        e = energy * 0 + np.e
        exp_term = torch.pow(e, -2.77 * energy / 2)
        return diff, energy, exp_term

    # target takes values in 1 (good), -1 (bad) so (1-target)/2 is 0 for good pairs and 1 for bad ones, (1+target) / 2 inverse
    def forward(self, input1, input2, y):
        _, energy, exp_term = self.terms(input1, input2, y)
        loss_g = (1 + y) * energy * energy / 2
        loss_i = (1 - y) * 2 * exp_term
        loss = (loss_g + loss_i).sum()
        if self.size_average:
            loss = loss / y.size(0)
        self.save_for_backward(input1, input2, y)
        return torch.Tensor([loss])

    def backward(self, grad_output):
        input1, input2, y = self.saved_tensors
        diff, energy, exp_term = self.terms(input1, input2, y)
        # represents the derivative w.r.t. input1 of energy
        diff[diff.lt(0)] = -1
        diff[diff.ge(0)] = 1
        y_g = (1 + y).view(-1, 1).expand_as(input1)
        y_i = (1 - y).view(-1, 1).expand_as(input1)
        energy = energy.expand_as(input1)
        exp_term = exp_term.expand_as(input1)
        grad1 = y_g * diff * energy - 2.77 * y_i * diff * exp_term
        grad2 = -grad1
        if self.size_average:
            grad1.div_(y.size(0))
            grad2.div_(y.size(0))
        return grad1, grad2, None


class MetricLoss(nn.Module):

    def __init__(self, size_average=True):
        super(MetricLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return MetricL(self.size_average)(input1, input2, target)
