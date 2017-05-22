import torch.nn as nn
import torchvision.models as models


# n < 0 sets all modules/blocks to be untrained
def set_untrained_blocks(containers, n):
    # first make sure everything is trainable (not trainable if n<0)
    for container in containers:
        for m in container:
            for p in m.parameters():
                p.requires_grad = n >= 0

    count = 0
    for seq in containers:
        for m in seq:
            if count >= n:
                break
            if sum(1 for _ in m.parameters()) <= 0:
                # exclude modules without params from count
                continue
            for p in m.parameters():
                p.requires_grad = False
            count += 1


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
        conv.weight.data[i] = fc.weight.data[i].view(in_channels, *in_size2d).clone()
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


def copy_bn_params(m, base_m):
    if m.weight is not None:
        m.weight.data.copy_(base_m.weight.data)
    if m.bias is not None:
        m.bias.data.copy_(base_m.bias.data)
    m.running_mean.copy_(base_m.running_mean)
    m.running_var.copy_(base_m.running_var)


def copy_bn_all(seq, base_seq):
    for m, base_m in zip(seq, base_seq):
        if isinstance(m, nn.Sequential):
            copy_bn_all(m, base_m)
        if isinstance(m, nn.BatchNorm2d):
            copy_bn_params(m, base_m)
        if isinstance(m, models.resnet.BasicBlock):
            copy_bn_params(m.bn1, base_m.bn1)
            copy_bn_params(m.bn2, base_m.bn2)
            if m.downsample is None:
                continue
            copy_bn_all(m.downsample, base_m.downsample)
        if isinstance(m, models.resnet.Bottleneck):
            copy_bn_params(m.bn1, base_m.bn1)
            copy_bn_params(m.bn2, base_m.bn2)
            copy_bn_params(m.bn3, base_m.bn3)
            if m.downsample is None:
                continue
            copy_bn_all(m.downsample, base_m.downsample)


def bn_new_params(bn, **kwargs):
    w, b, rm, rv = bn.weight, bn.bias, bn.running_mean, bn.running_var
    new_bn = nn.BatchNorm2d(bn.num_features, **kwargs)
    if w and new_bn.weight:
        new_bn.weight.data = w.data.clone()
    if b and new_bn.bias:
        new_bn.bias.data = b.data.clone()
    new_bn.running_mean = rm.clone()
    new_bn.running_var = rv.clone()
    return new_bn


def set_batch_norm_params(seq, **kwargs):
    for name, block in seq._modules.items():
        if isinstance(block, nn.Sequential):
            set_batch_norm_params(block, **kwargs)
        if isinstance(block, nn.BatchNorm2d):
            seq._modules[name] = bn_new_params(block, **kwargs)
        if isinstance(block, models.resnet.BasicBlock):
            block.bn1 = bn_new_params(block.bn1, **kwargs)
            block.bn2 = bn_new_params(block.bn2, **kwargs)
            if block.downsample is None:
                continue
            set_batch_norm_params(block.downsample, **kwargs)
        if isinstance(block, models.resnet.Bottleneck):
            block.bn1 = bn_new_params(block.bn1, **kwargs)
            block.bn2 = bn_new_params(block.bn2, **kwargs)
            block.bn3 = bn_new_params(block.bn3, **kwargs)
            if block.downsample is None:
                continue
            set_batch_norm_params(block.downsample, **kwargs)


def set_batch_norm_train(seq, train):
    for block in seq:
        if isinstance(block, nn.Sequential):
            set_batch_norm_train(block, train)
        if isinstance(block, nn.BatchNorm2d):
            block.train(mode=train)
        if isinstance(block, models.resnet.BasicBlock):
            block.bn1.train(mode=train)
            block.bn2.train(mode=train)
            if block.downsample is None:
                continue
            set_batch_norm_train(block.downsample, train)
        if isinstance(block, models.resnet.Bottleneck):
            block.bn1.train(mode=train)
            block.bn2.train(mode=train)
            block.bn3.train(mode=train)
            if block.downsample is None:
                continue
            set_batch_norm_train(block.downsample, train)


# net is assumed to have only one component containing BatchNorm modules:
# net.features
def set_net_train(net, train, bn_train=False):
    net.train(mode=train)
    if train and not bn_train:
        set_batch_norm_train(net.features, False)
