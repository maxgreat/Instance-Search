# -*- encoding: utf-8 -*-

import traceback
import torch.optim as optim

from os import path

from model.siamese import *
from train_classif import *
from train_siam import *
from dataset import ReadImages
from test_params import P
from utils import imread_rgb


def get_class_net(labels):
    if P.finetuning:
        net = TuneClassif(P.cnn_model(pretrained=True), len(labels), untrained_blocks=P.untrained_blocks, reduc=P.classif_feature_reduc)
    else:
        net = P.cnn_model()
    if P.classif_preload_net:
        net.load_state_dict(torch.load(P.classif_preload_net))
    net = move_device(net, P.cuda_device)
    return net


def get_feature_net(class_net):
    if P.feature_net_use_class_net:
        net = FeatureNet(class_net, P.feature_size2d, P.feature_net_average, P.feature_net_classify)
    else:
        net = FeatureNet(P.cnn_model(pretrained=P.finetuning), P.feature_size2d, P.feature_net_average, P.feature_net_classify)
    net = move_device(net, P.cuda_device)
    return net


def get_siamese_net(feature_net):
    if P.siam_model == 'siam2' and P.siam_use_feature_net:
        net = Siamese2(feature_net, P.siam2_k, P.siam_feature_dim, P.feature_size2d)
    elif P.siam_model == 'siam2':
        net = Siamese2(P.cnn_model(pretrained=P.finetuning), P.siam2_k, P.siam_feature_dim, P.feature_size2d)
    elif P.siam_use_feature_net:
        net = Siamese1(feature_net, P.siam_feature_dim, P.feature_size2d, P.siam_conv_average)
    else:
        net = Siamese1(P.cnn_model(pretrained=P.finetuning), P.siam_feature_dim, P.feature_size2d, P.siam_conv_average)
    if P.siam_preload_net:
        net.load_state_dict(torch.load(P.siam_preload_net))
    net = move_device(net, P.cuda_device)
    return net


def classif(labels, classif_test_train_set, classif_test_set, classif_train_set):
    class_net = get_class_net(labels)

    optimizer = optim.SGD((p for p in class_net.parameters() if p.requires_grad), lr=P.classif_lr, momentum=P.classif_momentum, weight_decay=P.classif_weight_decay)
    criterion = nn.CrossEntropyLoss(size_average=P.classif_loss_avg)
    testset_tuple = (classif_test_set, classif_test_train_set)
    if P.classif_test_upfront:
        P.log('Upfront testing of classification model')
        score = test_print_classif(class_net, testset_tuple, labels)
    else:
        score = 0
    if P.classif_train:
        P.log('Starting classification training')
        # TODO try normal weight initialization in classification training (see faster rcnn in pytorch)
        train_classif(class_net, classif_train_set, testset_tuple, labels, criterion, optimizer, best_score=score)
        P.log('Finished classification training')
    return class_net


def siam(feature_net, labels, siam_test_set, siam_test_train_set, siam_train_set):
    net = get_siamese_net(feature_net)

    optimizer = optim.SGD((p for p in net.parameters() if p.requires_grad), lr=P.siam_lr, momentum=P.siam_momentum, weight_decay=P.siam_weight_decay)
    # criterion = nn.CosineEmbeddingLoss(margin=P.siam_cos_margin, size_average=P.siam_loss_avg)
    if P.siam_train_mode == 'couples':
        criterion = nn.CosineEmbeddingLoss(margin=P.siam_cos_margin, size_average=P.siam_loss_avg)
    else:
        criterion = TripletLoss(margin=P.siam_triplet_margin, size_average=P.siam_loss_avg)
    if P.siam_double_objective:
        criterion2 = nn.CrossEntropyLoss(size_average=P.siam_do_loss2_avg)
    else:
        criterion2 = None
    testset_tuple = (siam_test_set, siam_test_train_set)
    if P.siam_test_upfront:
        P.log('Upfront testing of Siamese net')
        score = test_print_siamese(net, testset_tuple)
    else:
        score = 0
    if P.siam_train_mode == 'couples':
        f = train_siam_couples
    elif P.siam_choice_mode == 'easy-hard':
        f = train_siam_triplets
    else:
        f = train_siam_triplets_pos_couples
    if P.siam_train:
        P.log('Starting descriptor training')
        f(net, siam_train_set, testset_tuple, labels, criterion, optimizer, criterion2, best_score=score)
        P.log('Finished descriptor training')


def main():
    # training and test sets (scaled to 300 on the small side)
    train_set_full = ReadImages.readImageswithPattern(
        P.dataset_full, P.dataset_match_img)
    test_set_full = ReadImages.readImageswithPattern(
        P.dataset_full + '/test', P.dataset_match_img)

    # define the labels list
    labels_list = [t[1] for t in train_set_full if 'wall' not in t[1]]
    labels = list(set(labels_list))  # we have to give a number for each label

    P.log('Loading and transforming train/test sets.')

    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    def trans_dataset(dataset, pre_procs, trans, filters=None):
        if filters is None:
            filters = [lambda x, y: True for _ in pre_procs]
        outs = [[] for _ in pre_procs]
        for im, lab in dataset:
            im_o = imread_rgb(im)
            for out, pre_proc, t, f in zip(outs, pre_procs, trans, filters):
                if f(im, lab):
                    im_out = t(im_o) if pre_proc else im_o
                    out.append((im_out, lab))
        return outs

    train_pre_procs = [P.classif_train_pre_proc, P.classif_test_pre_proc, P.siam_train_pre_proc, P.siam_test_pre_proc]
    train_trans = [P.classif_train_trans, P.classif_test_trans, P.siam_train_trans, P.siam_test_trans]

    classif_train_set, classif_test_train_set, siam_train_set, siam_test_train_set = trans_dataset(train_set_full, train_pre_procs, train_trans)

    test_pre_procs = [P.classif_test_pre_proc, P.siam_test_pre_proc]
    test_trans = [P.classif_test_trans, P.siam_test_trans]
    test_filters = [lambda im, lab: lab in labels for _ in test_trans]
    classif_test_set, siam_test_set = trans_dataset(test_set_full, test_pre_procs, test_trans, test_filters)

    class_net = classif(labels, classif_test_train_set, classif_test_set, classif_train_set)
    feature_net = get_feature_net(class_net)
    if P.feature_net_upfront:
        P.log('Upfront testing of class/feature net as global descriptor')
        test_print_siamese(feature_net, (siam_test_set, siam_test_train_set))
    siam(feature_net, labels, siam_test_set, siam_test_train_set, siam_train_set)


if __name__ == '__main__':
    with torch.cuda.device(P.cuda_device):
        try:
            main()
        except:
            P.log_detail(None, traceback.format_exc())
            raise
