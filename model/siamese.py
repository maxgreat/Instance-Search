# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from nn_utils import *
from custom_modules import *


class TuneClassif(nn.Module):
    """
        Image classification network based on a pretrained network
        which is then finetuned to a different dataset
        It's assumed that the last layer of the given network
        is a fully connected (linear) one
        untrained specifies how many layers or blocks of layers are
        left untrained (only layers with parameters are counted). for ResNet, each 'BottleNeck' or 'BasicBlock' (block containing skip connection for residual) is considered as one block
    """

    def __init__(self, net, num_classes, untrained=-1, reduc=True):
        super(TuneClassif, self).__init__()
        self.features, self.feature_reduc, self.classifier = extract_layers(net)
        # make sure we never retrain the first few layers
        # this is usually not needed
        set_untrained_blocks([self.features, self.classifier], untrained)

        # replace last module of classifier with a reduced one
        last_module = self.classifier[len(self.classifier._modules) - 1]
        if not isinstance(last_module, nn.Linear) or last_module.out_features != num_classes:
            for name, module in self.classifier._modules.items():
                if module is last_module:
                    self.classifier._modules[name] = nn.Linear(module.in_features, num_classes)

        self.feature_size = num_classes
        # if no reduc is wanted, remove it
        if not reduc:
            factor = 1
            for m in self.feature_reduc:
                try:
                    factor *= (m.kernel_size[0] * m.kernel_size[1])
                except TypeError:
                    factor *= m.kernel_size * m.kernel_size
            # increase the number of input features on first classifier module
            for name, module in self.classifier._modules.items():
                if module is self.classifier[0]:
                    self.classifier._modules[name] = nn.Linear(module.in_features * factor, module.out_features)
            self.feature_reduc = nn.Sequential()

    def forward(self, x):
        x = self.features(x)
        x = self.feature_reduc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TuneClassifSub(TuneClassif):
    """
        Image classification network based on a pretrained network
        which is then finetuned to a different dataset, as above
        Here, all sub-parts of the image are classified by
        convolutionalizing the linear classification layers
    """
    def __init__(self, net, num_classes, feature_size2d, untrained=-1):
        super(TuneClassifSub, self).__init__(net, num_classes, untrained, reduc=True)
        reduc_count = sum(1 for _ in self.feature_reduc)
        if reduc_count > 0:
            # in a ResNet, apply stride 1 feature size avg pool reduction
            self.feature_reduc = nn.Sequential(
                nn.AvgPool2d(feature_size2d, stride=1)
            )
        # convolutionalize the linear layers in classifier
        count = 0
        for name, module in self.classifier._modules.items():
            if isinstance(module, nn.modules.linear.Linear):
                size2d = feature_size2d
                if reduc_count > 0 or count > 0:
                    size2d = (1, 1)
                self.classifier._modules[name] = convolutionalize(module, size2d)
                count += 1

    def forward_single(self, x):
        x = self.features(x)
        x = self.feature_reduc(x)
        x = self.classifier(x)
        return x

    def forward(self, *scales):
        return [self.forward_single(x) for x in scales]


class DescriptorNet(nn.Module):
    """
        Define a siamese network
        Given a network, obtain its features, then apply spatial reduction
        (optional) and a norm, shift+linear, norm reduction to obtain a
        descriptor.
        TODO description
    """
    def __init__(self, net, feature_dim, feature_size2d, untrained=-1):
        super(DescriptorNet, self).__init__()
        self.features, _, classifier = extract_layers(net)
        set_untrained_blocks([self.features], untrained)
        factor = feature_size2d[0] * feature_size2d[1]
        in_features = get_feature_size(self.features, factor)
        if feature_dim <= 0:
            self.feature_size = get_feature_size(classifier)
        else:
            self.feature_size = feature_dim
        self.feature_reduc1 = nn.Sequential(
            NormalizeL2(),
            Shift(in_features),
            nn.Linear(in_features, self.feature_size)
        )
        self.feature_reduc2 = NormalizeL2()

    def forward_single(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reduc1(x)
        x = self.feature_reduc2(x)
        return x

    def forward(self, x1, x2=None, x3=None):
        if self.training and x3 is not None:
            return self.forward_single(x1), self.forward_single(x2), self.forward_single(x3)
        elif self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)


class RegionDescriptorNet(nn.Module):
    """
        Define a siamese network
        Given a network, obtain its features and apply spatial reduction
        (optional). The feature maps can have any size here, so we apply
        a classifier (obtained from the given network) to all locations
        in the feature map. Finally, we sum the features in those regions
        obtaining the highest classification values and apply normalization,
        shifting, linear, normalization to obtain a global descriptor.
        In order to allow training for both the descriptor and the classifier,
        the classification values are output as well as the descriptor
        for all input images.

        Use the k highest values from the classifier to obtain descriptor
    """
    def __init__(self, net, k, feature_dim, feature_size2d, untrained=-1):
        super(RegionDescriptorNet, self).__init__()
        self.k = k
        self.feature_size2d = feature_size2d
        self.features, self.feature_reduc, self.classifier = extract_layers(net)

        # factor = 1
        factor = feature_size2d[0] * feature_size2d[1]
        in_features = get_feature_size(self.features, factor)
        if feature_dim <= 0:
            self.feature_size = get_feature_size(classifier)
        else:
            self.feature_size = feature_dim
        reduc_count = sum(1 for _ in self.feature_reduc)
        if reduc_count > 0:
            # we are a ResNet or similar, apply feature_size AvgPool stride 1
            self.feature_reduc = nn.Sequential(
                nn.AvgPool2d(feature_size2d, stride=1)
            )
        # convolutionalize the linear layers in classifier
        count = 0
        for name, module in self.classifier._modules.items():
            if isinstance(module, nn.modules.linear.Linear):
                size2d = feature_size2d
                if reduc_count > 0 or count > 0:
                    size2d = (1, 1)
                self.classifier._modules[name] = convolutionalize(module, size2d)
                count += 1
        set_untrained_blocks([self.features, self.classifier], untrained)
        self.feature_reduc1 = nn.Sequential(
            NormalizeL2(),
            Shift(in_features),
            nn.Linear(in_features, self.feature_size)
        )
        self.feature_reduc2 = NormalizeL2()

    # this can only be done using a single input (batch size: 1) TODO
    def forward_single(self, x):
        x = self.features(x)
        c = self.feature_reduc(x)
        c = self.classifier(c)
        # get maximal classification values and choose indexes with
        # highest maximal classification
        c_maxv, _ = c.max(1)
        c_maxv = c_maxv.view(-1)
        k = min(c_maxv.size(0), self.k)
        _, flat_idx = c_maxv.topk(k)

        # transform flat classification indexes to feature indexes
        # first, flat index -> 2d classification index, then add
        # feature size to obtain the region in feature map
        def feature_idx(flat_idx):
            cls_idx = flat_idx // c.size(3), flat_idx % c.size(3)
            return (cls_idx[0], cls_idx[0] + self.feature_size2d[0],
                    cls_idx[1], cls_idx[1] + self.feature_size2d[1])
        top_idx = [feature_idx(int(i)) for i in flat_idx.data]
        # needed for output
        tmp = c_maxv.data.clone().resize_(c.size(0), self.feature_size)
        acc = Variable(tmp.fill_(0))
        tmp = c_maxv.data.clone().resize_(c.size(0), c.size(1), self.k)
        cls_out = Variable(tmp.fill_(0))

        # for all top maximal classification indexes, output the actual
        # classification values at those indexes
        # for the descriptor, use the feature indexes and then reduce
        # accumulate regional descriptors using addition
        i = 0
        for x1, x2, y1, y2 in top_idx:
            cls_out[:, :, i] = c[:, :, x1, y1]
            i += 1
            region = x[:, :, x1:x2, y1:y2].contiguous().view(x.size(0), -1)
            region = self.feature_reduc1(region)
            acc = acc + region
        # finally, perform final reduction (normalization)
        x = self.feature_reduc2(acc)
        return x, cls_out

    def forward(self, x1, x2=None, x3=None):
        if self.training and x3 is not None:
            return self.forward_single(x1), self.forward_single(x2), self.forward_single(x3)
        elif self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)[0]
