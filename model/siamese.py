# -*- encoding: utf-8 -*-

import torch.nn as nn
import torchvision.models as models
from custom_modules import *


def get_feature_size(seq, factor=1, default=-1):
    feature_size = default
    for module in seq:
        if isinstance(module, models.resnet.Bottleneck):
            feature_size = module.conv3.out_channels * factor
        if isinstance(module, models.resnet.BasicBlock):
            feature_size = module.conv2.out_channels * factor
        if isinstance(module, nn.modules.Conv2d):
            feature_size = module.out_channels * factor
        if isinstance(module, nn.modules.linear.Linear):
            feature_size = module.out_features
    return feature_size


def extract_layers(net):
    if hasattr(net, 'features') and hasattr(net, 'feature_reduc') and hasattr(net, 'classifier'):
        return net.features, net.feature_reduc, net.classifier
    if isinstance(net, models.ResNet):
        features = [net.conv1, net.bn1, net.relu, net.maxpool]
        features.extend(net.layer1)
        features.extend(net.layer2)
        features.extend(net.layer3)
        features.extend(net.layer4)
        features = nn.Sequential(*features)
        feature_reduc = nn.Sequential(net.avgpool)
        classifier = nn.Sequential(net.fc)
    else:
        features, classifier = net.features, net.classifier
        feature_reduc = nn.Sequential()
    return features, feature_reduc, classifier


class FeatureNet(nn.Module):
    """
        A simple network consisting only of the features extracted
        from an underlying CNN, which can be averaged spatially and
        are then returned as a flat vector
    """
    def __init__(self, net, feature_size2d, average_features=False, classify=False):
        super(FeatureNet, self).__init__()
        self.features, self.feature_reduc, self.classifier = extract_layers(net)
        if not classify:
            self.feature_reduc, self.classifier = None, None
            factor = feature_size2d[0] * feature_size2d[1]
            self.feature_size = get_feature_size(self.features, factor)
            if average_features:
                self.feature_reduc = nn.AvgPool2d(feature_size2d)
                self.feature_size /= (feature_size2d[0] * feature_size2d[1])
        else:
            self.feature_size = get_feature_size(self.classifier)
        self.norm = NormalizeL2()

    def forward(self, x):
        x = self.features(x)
        if self.feature_reduc:
            x = self.feature_reduc(x)
        x = x.view(x.size(0), -1)
        if self.classifier:
            x = self.classifier(x)
        x = self.norm(x)
        return x


class TuneClassif(nn.Module):
    """
        Image classification network based on a pretrained network
        which is then finetuned to a different dataset
        It's assumed that the last layer of the given network
        is a fully connected (linear) one
        untrained_blocks specifies how many layers or blocks of layers are
        left untrained (only layers with parameters are counted). for ResNet, each 'BottleNeck' or 'BasicBlock' (block containing skip connection for residual) is considered as one block
    """

    def __init__(self, net, num_classes, untrained_blocks=-1):
        super(TuneClassif, self).__init__()
        self.features, self.feature_reduc, self.classifier = extract_layers(net)
        if untrained_blocks < 0:
            untrained_blocks = sum(1 for _ in self.features) + sum(1 for _ in self.classifier)
        # make sure we never retrain the first few layers
        # this is usually not needed
        seqs = [self.features, self.feature_reduc, self.classifier]

        def has_param(m):
            return sum(1 for _ in m.parameters()) > 0
        count = 0
        for module in (m for seq in seqs for m in seq if has_param(m)):
            if count >= untrained_blocks:
                break
            count += 1
            for p in module.parameters():
                p.requires_grad = False

        for name, module in self.classifier._modules.items():
            if module is self.classifier[len(self.classifier._modules) - 1]:
                self.classifier._modules[name] = nn.Linear(module.in_features, num_classes)
        self.feature_size = num_classes

    def forward(self, x):
        x = self.features(x)
        x = self.feature_reduc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Siamese1(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier
        TODO description, feature reduc (resnet was trained for this avgpool)
    """
    def __init__(self, net, feature_dim, feature_size2d, spatial_avg_factor=(1, 1)):
        super(Siamese1, self).__init__()
        self.features, _, classifier = extract_layers(net)
        if spatial_avg_factor[0] == -1:
            spatial_avg_factor[0] = feature_size2d[0]
        if spatial_avg_factor[1] == -1:
            spatial_avg_factor[1] = feature_size2d[1]
        self.spatial_feature_reduc = nn.Sequential(
            nn.AvgPool2d(spatial_avg_factor)
        )
        factor = feature_size2d[0] * feature_size2d[1] / (spatial_avg_factor[0] * spatial_avg_factor[1])
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
        x = self.spatial_feature_reduc(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reduc1(x)
        x = self.feature_reduc2(x)
        return x

    def forward(self, x1, x2=None, x3=None):
        if self.training and x3:
            return self.forward_single(x1), self.forward_single(x2), self.forward_single(x3)
        elif self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)
