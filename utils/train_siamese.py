# -*- encoding: utf-8 -*-

import torch
import random
from os import path
from model.nn_utils import set_net_train
from general import tensor_t
from general import log, save_uuid, unique_str
from dataset import get_pos_couples
from metrics import precision1, mean_avg_precision


# get byte tensors indicating the indexes of images having a different label
def get_lab_indicators(dataset, device):
    n = len(dataset)
    indicators = {}
    for _, lab1, _ in dataset:
        if lab1 in indicators:
            continue
        indicator = tensor_t(torch.ByteTensor, device, n).fill_(0)
        for i2, (_, lab2, _) in enumerate(dataset):
            if lab1 == lab2:
                indicator[i2] = 1
        indicators[lab1] = indicator
    return indicators


# determine the device where embeddings should be stored
# and the feature dimension for a descriptor
def embeddings_device_dim(P, net, n, sim_matrix=False):
    # get best device for embeddings (and possibly similarity matrix),
    # as well as the feature vector size.
    # usually, this is the configured cuda device.
    # but it could be CPU if embeddings/number of items are too large
    device = P.cuda_device
    out_size = P.feature_dim
    if hasattr(net, 'feature_size') and out_size <= 0:
        out_size = net.feature_size
    if n * out_size * 4 > P.embeddings_cuda_size:
        device = -1
    if sim_matrix and n * n * 4 > P.embeddings_cuda_size:
        device = -1
    return device, out_size


# get all similarities between pairs of images of the dataset
# net is assumed to be in train mode
def get_similarities(P, get_embeddings, net, dataset):
    set_net_train(net, False)
    n = len(dataset)
    d, o = embeddings_device_dim(P, net, n, sim_matrix=True)
    embeddings = get_embeddings(net, dataset, d, o)
    similarities = torch.mm(embeddings, embeddings.t())
    set_net_train(net, True, bn_train=P.train_bn)
    return similarities, d


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (precision at 1, not the average precision/ranking on the ref set). TODO
def test_descriptor_net(P, get_embeddings, net, test_set, test_ref_set, kth=1):
    d, o = embeddings_device_dim(P, net, max(len(test_set), len(test_ref_set)))
    test_embeddings = get_embeddings(net, test_set, d, o)
    ref_embeddings = get_embeddings(net, test_ref_set, d, o)

    # calculate all similarities as a simple matrix multiplication
    # since embeddings are assumed to be normalized
    # the similarities here should always be on CPU
    # (kthvalue is only implemented there and we don't need GPU perf)
    sim = torch.mm(test_embeddings, ref_embeddings.t()).cpu()
    # stats
    prec1, correct, total, max_sim, max_label = precision1(sim, test_set, test_ref_set, kth)
    mAP = mean_avg_precision(sim, test_set, test_ref_set, kth)
    sum_pos = sum(sim[i, j] for i, (_, test_label, _) in enumerate(test_set) for j, (_, ref_label, _) in enumerate(test_ref_set) if test_label == ref_label)
    sum_neg = sim.sum() - sum_pos
    sum_max = max_sim.sum()
    lab_dict = dict([(lab, {}) for _, lab, _ in test_set])
    for j, (_, lab, _) in enumerate(test_set):
        d = lab_dict[lab]
        lab = max_label[j]
        d.setdefault(lab, d.get(lab, 0) + 1)
    return prec1, correct, total, sum_pos, sum_neg, sum_max, mAP, lab_dict


# Generic function to test and print stats when training a descriptor net
def test_print_descriptor(train_type, P, net, testset_tuple, get_embeddings, best_score=0, epoch=0):
    def print_stats(prefix, p1, c, t, avg_pos, avg_neg, avg_max, mAP):
        s1 = 'Correct: {0} / {1} - acc: {2:.4f} - mAP:{3:.4f}\n'.format(c, t, p1, mAP)
        s2 = 'AVG cosine sim (sq dist) values: pos: {0:.4f} ({1:.4f}), neg: {2:.4f} ({3:.4f}), max: {4:.4f} ({5:.4f})'.format(avg_pos, 2 - 2 * avg_pos, avg_neg, 2 - 2 * avg_neg, avg_max, 2 - 2 * avg_max)
        log(P, prefix + s1 + s2)

    test_set, test_ref_set = testset_tuple
    set_net_train(net, False)
    prec1, correct, tot, sum_pos, sum_neg, sum_max, mAP, lab_dict = test_descriptor_net(P, get_embeddings, net, test_set, test_ref_set)
    # can save labels dictionary (predicted labels for all test labels)
    # TODO

    num_pos = sum(test_label == ref_label for _, test_label, _ in test_set for _, ref_label, _ in test_ref_set)
    num_neg = len(test_set) * len(test_ref_set) - num_pos

    if (correct > best_score):
        best_score = correct
        prefix = '{0}, EPOCH:{1}, SCORE:{2}'.format(train_type, epoch, correct)
        save_uuid(P, prefix)
        torch.save(net.state_dict(), path.join(P.save_dir, unique_str(P) + "_best_siam.pth.tar"))
    print_stats('TEST - ', prec1, correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(test_set), mAP)
    torch.save(net.state_dict(), path.join(P.save_dir, "model_siam_" + str(epoch) + ".pth.tar"))

    # training set accuracy (choose second highest value,
    # as highest should almost certainly be the same image)
    # choose train samples with at least 2 other images for the query
    couples = get_pos_couples(test_ref_set)
    train_test_set = random.sample(test_ref_set, max(1, len(test_ref_set) // 10))
    train_test_set = filter(lambda x: len(couples[x[1]]) >= 3, train_test_set)
    prec1, correct, tot, sum_pos, sum_neg, sum_max, mAP, _ = test_descriptor_net(P, get_embeddings, net, train_test_set, test_ref_set, kth=2)
    num_pos = sum(test_label == ref_label for _, test_label, _ in train_test_set for _, ref_label, _ in test_ref_set)
    num_neg = len(train_test_set) * len(test_ref_set) - num_pos
    print_stats('TRAIN - ', prec1, correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(train_test_set), mAP)
    set_net_train(net, True, bn_train=P.train_bn)
    return best_score
