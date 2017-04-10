import torch.nn as nn
import torchvision.models as models


def convolutionalize(fc, in_size2d):
    # Turn an FC layer into Conv2D layer by copying weights the right way
    out_size = fc.out_features
    in_size_total = fc.in_features
    if in_size_total % (in_size2d[0] * in_size2d[1]) != 0:
        raise ValueError('FC in_feature size {0} is not divisible by in_size2d {1}'.format(in_size_total, in_size2d))
    in_channels = in_size_total // (in_size2d[0] * in_size2d[1])
    has_bias = fc.bias is not None
    conv = nn.Conv2d(in_channels, out_size, in_size2d, bias=has_bias)
    if has_bias:
        conv.bias.data = fc.bias.data.clone()
    for i in range(out_size):
        # TODO check if this is correct
        conv.weight.data = fc.weight.data[i].view(in_channels, *in_size2d).clone()
    return conv


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
