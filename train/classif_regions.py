# -*- encoding: utf-8 -*-

import traceback
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from classif_regions_p import P
from utils import move_device, tensor_t, tensor, fold_batches, train_gen
from utils import imread_rgb, log, log_detail, test_print_classif
from utils import test_print_descriptor, get_images_labels
from model.siamese import TuneClassif, TuneClassifSub
from model.custom_modules import NormalizeL2Fun

# keep labels as global variable. they are initialized after
# train set has been loaded and then kept constant
labels = []
train_type = P.cnn_model.lower() + ' Classification sub-regions'


# test a classifier model. it should be in eval mode
def test_classif_net(net, test_set):
    """
        Test the network accuracy on a test_set
        Return the number of success and the number of evaluations done
    """
    trans = P.test_trans
    if P.test_pre_proc:
        trans = transforms.Compose([])

    def eval_batch_test(last, i, is_final, batch):
        correct, total = last
        im_trans = trans(batch[0][0])
        test_in = move_device(im_trans.unsqueeze(0), P.cuda_device)
        out = net(Variable(test_in, volatile=True))[0].data
        # first get all maximal values for classification
        # then, use the spatial region with the highest maximal value
        # to make a prediction
        max_pred, predicted = torch.max(out, 1)
        _, max_subp = torch.max(max_pred.view(-1), 0)
        predicted = predicted.view(-1)[max_subp[0]]
        total += 1
        correct += (labels.index(batch[0][1]) == predicted)

        return correct, total

    # batch size has to be 1 here
    return fold_batches(eval_batch_test, (0, 0), test_set, 1)


def train_classif_subparts(net, train_set, testset_tuple, criterion, optimizer, best_score=0):
    # trans is a list of transforms for each scale here
    trans_scales = P.train_trans
    for i, t in enumerate(trans_scales):
        if P.train_pre_proc[i]:
            trans_scales[i] = transforms.Compose([])

    # images are already pre-processed in all cases
    def create_epoch(epoch, train_set, testset_tuple):
        random.shuffle(train_set)
        # labels are needed for stats
        return train_set, {}

    def create_batch(batch, n):
        # must proceed image by image (since different input sizes)
        # each image/batch is composed of multiple scales
        n_sc = len(batch[0][0])
        train_in_scales = []
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, 1)
        labels_in.fill_(labels.index(batch[0][1]))
        for j in range(n_sc):
            im = trans_scales[j](batch[0][0][j])
            train_in = move_device(im.unsqueeze(0), P.cuda_device)
            train_in_scales.append(train_in)
        return train_in_scales, [labels_in]

    def create_loss(scales_out, labels_list):
        # scales_out is a list over all scales,
        # with all sub-region classifications for each scale
        labels_in = labels_list[0]
        loss = None
        for s, t_out in enumerate(scales_out):
            # batch size is 1, only consider this output
            t_out0 = t_out[0]
            # all spatial outputs are of shape (num_classes, width, height)
            # make a 'batch' as follows: (width * height, num_classes)
            # then apply loss to the whole batch, and accumulate over scales
            t_out_all = t_out0.view(t_out0.size(0), -1).t()
            if loss is None:
                loss = criterion(t_out_all, labels_in.expand(t_out_all.size(0)))
            else:
                loss += criterion(t_out_all, labels_in.expand(t_out_all.size(0)))
        if P.train_loss_avg:
            loss /= len(scales_out)
        return loss, None

    train_gen(train_type, P, test_print_classif, test_classif_net, net,
              train_set, testset_tuple, optimizer, create_epoch, create_batch,
              create_loss, best_score=best_score)


# get the embeddings as the normalized output of the classification
# values where the highest maximal activation occurred
def get_embeddings(net, dataset, device, out_size):
    test_trans = P.test_trans
    if P.test_pre_proc:
        test_trans = transforms.Compose([])

    def batch(last, i, is_final, batch):
        embeddings = last
        im_trans = test_trans(batch[0][0])
        test_in = move_device(im_trans.unsqueeze(0), P.cuda_device)
        out = net(Variable(test_in, volatile=True))[0].data
        # first, determine location of highest maximal activation
        max_pred, _ = out.max(1)
        max_pred1, max_i1 = max_pred.max(2)
        _, max_i2 = max_pred1.max(3)
        i2 = max_i2.view(-1)[0]
        i1 = max_i1.view(-1)[i2]
        # we have the indexes of the highest maximal activation,
        # get the classification values at this point and normalize
        out = out[:, :, i1, i2]
        out = NormalizeL2Fun()(Variable(out, volatile=True))
        out = out.data
        embeddings[i] = out[0]
        return embeddings

    init = tensor(device, len(dataset), out_size)
    return fold_batches(batch, init, dataset, 1)


def get_class_net():
    model = models.alexnet
    if P.cnn_model.lower() == 'resnet152':
        model = models.resnet152
    if P.bn_model:
        bn_model = TuneClassif(model(), len(labels))
        bn_model.load_state_dict(torch.load(P.bn_model, map_location=lambda storage, location: storage.cpu()))
        # copy_bn_all(net.features, bn_model.features)
    else:
        bn_model = model(pretrained=True)
    net = TuneClassifSub(bn_model, len(labels), P.feature_size2d, untrained=P.untrained_blocks)
    if P.preload_net:
        net.load_state_dict(torch.load(P.preload_net, map_location=lambda storage, location: storage.cpu()))
    net = move_device(net, P.cuda_device)
    return net


def main():
    # training and test sets
    train_set_full = get_images_labels(P.dataset_full, P.match_labels)
    test_set_full = get_images_labels(P.dataset_full + '/test', P.match_labels)

    labels_list = [t[1] for t in train_set_full]
    # we have to give a number to each label,
    # so we need a list here for the index
    labels.extend(sorted(list(set(labels_list))))

    log(P, 'Loading and transforming train/test sets.')

    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    train_set, test_train_set, test_set = [], [], []
    train_pre_f = [t if pre_proc else transforms.Compose([]) for t, pre_proc in zip(P.train_trans, P.train_pre_proc)]
    test_pre_f = P.test_trans if P.test_pre_proc else transforms.Compose([])
    train_scales = P.train_sub_scales
    for im, lab in train_set_full:
        im_o = imread_rgb(im)
        scales = [t(im_o) for t in train_scales]
        train_set.append((scales, lab, im))
        for j, t in enumerate(train_pre_f):
            scales[j] = t(scales[j])
        im_pre_test = test_pre_f(im_o) if test_pre_f else im_o
        test_train_set.append((im_pre_test, lab, im))

    for im, lab in test_set_full:
        if lab not in labels:
            continue
        im_o = imread_rgb(im)
        test_set.append((test_pre_f(im_o), lab, im))

    class_net = get_class_net()
    optimizer = optim.SGD((p for p in class_net.parameters() if p.requires_grad), lr=P.train_lr, momentum=P.train_momentum, weight_decay=P.train_weight_decay)
    criterion = nn.CrossEntropyLoss(size_average=P.train_loss_avg)
    testset_tuple = (test_set, test_train_set)
    if P.test_upfront:
        log(P, 'Upfront testing of classification model')
        score = test_print_classif(train_type, P, class_net, testset_tuple, test_classif_net)
    else:
        score = 0
    if P.train:
        log(P, 'Starting classification training')
        train_classif_subparts(class_net, train_set, testset_tuple, criterion, optimizer, best_score=score)
        log(P, 'Finished classification training')
    if P.test_descriptor_net:
        log(P, 'Testing as descriptor')
        test_print_descriptor(train_type, P, class_net, testset_tuple, get_embeddings)


if __name__ == '__main__':
    with torch.cuda.device(P.cuda_device):
        try:
            main()
        except:
            log_detail(P, None, traceback.format_exc())
            raise
