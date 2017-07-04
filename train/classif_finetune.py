# -*- encoding: utf-8 -*-

import traceback
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from classif_finetune_p import P
from utils import move_device, tensor_t, tensor, fold_batches, train_gen
from utils import imread_rgb, log, log_detail, test_print_classif
from utils import test_print_descriptor, get_images_labels
from model.siamese import TuneClassif
from model.custom_modules import NormalizeL2Fun

# keep labels as global variable. they are initialized after
# train set has been loaded and then kept constant
labels = []
train_type = P.cnn_model.lower() + ' Classification simple fine-tuning'


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
        n = len(batch)
        test_in = tensor(P.cuda_device, n, *P.image_input_size)
        for j, (testIm, _, _) in enumerate(batch):
            test_in[j] = trans(testIm)
        out = net(Variable(test_in, volatile=True)).data
        # first get all maximal values for classification
        # then, use the spatial region with the highest maximal value
        # to make a prediction
        _, predicted = torch.max(out, 1)
        total += n
        correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel, _) in enumerate(batch))
        return correct, total

    # batch size has to be 1 here
    return fold_batches(eval_batch_test, (0, 0), test_set, P.test_batch_size)


def train_classif(net, train_set, testset_tuple, criterion, optimizer, best_score=0):
    # trans is a list of transforms for each scale here
    trans = P.train_trans
    if P.train_pre_proc:
        trans = transforms.Compose([])

    # images are already pre-processed in all cases
    def create_epoch(epoch, train_set, testset_tuple):
        random.shuffle(train_set)
        # labels are needed for stats
        return train_set, {}

    def create_batch(batch, n):
        train_in = tensor(P.cuda_device, n, *P.image_input_size)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n)
        for j, (im, lab, _) in enumerate(batch):
            train_in[j] = trans(im)
            labels_in[j] = labels.index(lab)
        return [train_in], [labels_in]

    def create_loss(t_out, labels_list):
        return criterion(t_out, labels_list[0]), None

    train_gen(train_type, P, test_print_classif, test_classif_net, net,
              train_set, testset_tuple, optimizer, create_epoch, create_batch,
              create_loss, best_score=best_score)


# get the embeddings as the normalized output of the classification
def get_embeddings(net, dataset, device, out_size):
    trans = P.test_trans
    if P.test_pre_proc:
        trans = transforms.Compose([])

    if not P.embeddings_classify:
        # remove classifier and add back later
        classifier = net.classifier
        net.classifier = nn.Sequential()

    def batch(last, i, is_final, batch):
        embeddings = last
        n = len(batch)
        test_in = tensor(P.cuda_device, n, *P.image_input_size)
        for j, (testIm, _, _) in enumerate(batch):
            test_in[j] = trans(testIm)
        out = net(Variable(test_in, volatile=True))
        # we have the classification values. just normalize
        out = NormalizeL2Fun()(out)
        out = out.data
        for j in range(n):
            embeddings[i + j] = out[j]
        return embeddings

    init = tensor(device, len(dataset), out_size)
    embeddings = fold_batches(batch, init, dataset, P.test_batch_size)
    if not P.embeddings_classify:
        net.classifier = classifier
    return embeddings


def get_class_net():
    model = models.alexnet
    if P.cnn_model.lower() == 'resnet152':
        model = models.resnet152
    net = TuneClassif(model(pretrained=True), len(labels), untrained=P.untrained_blocks)
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
    train_pre_f = P.train_trans if P.train_pre_proc else transforms.Compose([])
    test_pre_f = P.test_trans if P.test_pre_proc else transforms.Compose([])
    for im, lab in train_set_full:
        im_o = imread_rgb(im)
        train_set.append((train_pre_f(im_o), lab, im))
        test_train_set.append((test_pre_f(im_o), lab, im))

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
        train_classif(class_net, train_set, testset_tuple, criterion, optimizer, best_score=score)
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
