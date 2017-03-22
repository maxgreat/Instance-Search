# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import numpy as np


# function to shift an input with a trainable parameter
class ShiftFun(Function):

    def __init__(self):
        super(ShiftFun, self).__init__()

    def forward(self, input, param):
        self.save_for_backward(input, param)
        return input + param.view(1, -1).expand_as(input)

    def backward(self, grad_output):
        input, param = self.saved_tensors
        grad_input = grad_output.clone()
        buf = param.clone().resize_(input.size(0)).fill_(1)
        grad_param = torch.mv(grad_output.t(), buf)
        return grad_input, grad_param


class Shift(nn.Module):

    def __init__(self, n_features):
        super(Shift, self).__init__()
        self.param = Parameter(torch.Tensor(n_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.param.data.fill_(0)

    def forward(self, input):
        return ShiftFun()(input, self.param)


# autograd function to normalize an input over the rows
# (each vector of a batch is normalized)
# the backward step follows the implementation of
# torch.legacy.nn.Normalize closely
class NormalizeL2Fun(Function):

    def __init__(self, eps=1e-10):
        super(NormalizeL2Fun, self).__init__()
        self.eps = eps

    def forward(self, input):
        self.save_for_backward(input)
        self.norm2 = input.pow(2).sum(1).add_(self.eps)
        self.norm = self.norm2.pow(0.5)
        output = input / self.norm.expand_as(input)
        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        gradInput = self.norm2.expand_as(input) * grad_output
        cross = (input * grad_output).sum(1)
        buf = input * cross.expand_as(input)
        gradInput.add_(-1, buf)
        cross = self.norm2 * self.norm
        gradInput.div_(cross.expand_as(gradInput))
        return gradInput


class NormalizeL2(nn.Module):

    def __init__(self):
        super(NormalizeL2, self).__init__()

    def forward(self, input):
        return NormalizeL2Fun()(input)


# metric loss according to Chopra et al "Learning a Similarity Metric Discriminatively, with Application to Face Verification"
# since we assume normalized vectors, we use Q=2
class MetricLossFun(Function):

    def __init__(self, size_average=True):
        super(MetricLossFun, self).__init__()
        self.size_average = size_average

    # TODO: more things could be done inplace
    # this is difficult and probs unnecessary though
    def terms(self, input1, input2, y):
        diff = input1 - input2
        energy = diff.norm(1, 1)
        e = (energy * 0).add_(np.e)  # fill with e, same shape as energy
        exp_term = e.pow_((-2.77 * energy).div_(2))
        return diff, energy, exp_term

    # target takes values in 1 (good), -1 (bad) so (1-target)/2 is 0 for good pairs and 1 for bad ones, (1+target) / 2 inverse
    def forward(self, input1, input2, y):
        self.save_for_backward(input1, input2, y)
        _, energy, exp_term = self.terms(input1, input2, y)
        loss = energy.mul_(energy).mul_(1 + y).div_(2)
        loss.add_(exp_term.mul_(1 - y).mul_(2))
        loss = loss.sum(0).view(1)
        if self.size_average:
            loss.div_(y.size(0))
        return loss

    def backward(self, grad_output):
        input1, input2, y = self.saved_tensors
        diff, energy, exp_term = self.terms(input1, input2, y)
        diff[diff.lt(0)] = -1
        diff[diff.ge(0)] = 1
        energy = energy.expand_as(input1)
        exp_term = exp_term.expand_as(input1)
        y_g = (1 + y).view(-1, 1).expand_as(input1)
        y_i = (1 - y).view(-1, 1).expand_as(input1)
        y_g = y_g.mul(diff).mul_(energy)
        y_i = y_i.mul(2.77).mul_(diff).mul_(exp_term)
        grad1 = y_g.add_(-1, y_i)
        grad2 = -grad1
        if self.size_average:
            grad1.div_(y.size(0))
            grad2.div_(y.size(0))
        g = grad_output[0]
        if g != 1:
            grad1.mul_(g)
            grad2.mul_(g)
        return grad1, grad2, None


class MetricLoss(nn.Module):

    def __init__(self, size_average=True):
        super(MetricLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return MetricLossFun(self.size_average)(input1, input2, target)


class TripletLossFun(Function):

    def __init__(self, margin, size_average=True, normalized=True):
        super(TripletLossFun, self).__init__()
        self.size_average = size_average
        self.margin = margin
        self.normalized = normalized

    # calculate for each sample i:
    # 1/2 (||anchor_i - pos_i||^2 - ||anchor_i - neg_i||^2 + 2margin)
    # then clamp to positive values and sum over all samples
    # when normalized, ||x1-x2||^2 = 2 - 2x1.x2
    # so the loss for i becomes: anchor_i . neg_i - anchor_i . pos_i + margin
    def forward(self, anchor, pos, neg):
        self.save_for_backward(anchor, pos, neg)
        if self.normalized:
            loss = (anchor * neg).sum(1)
            loss.add_(-1, (anchor * pos).sum(1))
            loss.add_(self.margin)
        else:
            sqdiff_pos = (anchor - pos).pow_(2)
            sqdiff_neg = (anchor - neg).pow_(2)
            loss = sqdiff_pos.sum(1)
            loss.add_(-1, sqdiff_neg.sum(1))
            loss.add_(self.margin * 2)
            loss.div_(2)
        self.clamp = torch.le(loss, 0)
        loss[self.clamp] = 0
        loss = loss.sum(0).view(1)
        if self.size_average:
            loss.div_(anchor.size(0))
        return loss

    def backward(self, grad_output):
        # grad_pos = -(anchor_i - pos_i) for sample i
        # grad_neg = (anchor_i - neg_i)
        # grad_anchor = (anchor_i - pos_i) - (anchor_i - neg_i)
        # = (neg_i - pos_i)
        # if normalized: grad_pos = -anchor_i, grad_neg = anchor_i
        # grad_anchor = neg_i - pos_i
        anchor, pos, neg = self.saved_tensors
        if self.normalized:
            grad_anchor = neg - pos
            grad_pos = -anchor
            grad_neg = -grad_pos
        else:
            grad_anchor = neg - pos
            grad_pos = pos - anchor
            grad_neg = anchor - neg
        c = self.clamp.expand_as(anchor)
        grad_anchor[c] = 0
        grad_pos[c] = 0
        grad_neg[c] = 0

        if self.size_average:
            grad_anchor.div_(anchor.size(0))
            grad_pos.div_(anchor.size(0))
            grad_neg.div_(anchor.size(0))
        g = grad_output[0]
        if g != 1:
            grad_anchor = grad_anchor.mul_(g)
            grad_pos = grad_pos.mul_(g)
            grad_neg = grad_neg.mul_(g)
        return grad_anchor, grad_pos, grad_neg


class TripletLoss(nn.Module):

    def __init__(self, margin, size_average=True, normalized=True):
        super(TripletLoss, self).__init__()
        self.size_average = size_average
        self.margin = margin
        self.normalized = normalized

    def forward(self, anchor, pos, neg):
        return TripletLossFun(self.margin, self.size_average, self.normalized)(anchor, pos, neg)
