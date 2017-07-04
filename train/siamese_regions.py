# -*- encoding: utf-8 -*-

import traceback
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from siamese_regions_p import P
from utils import move_device, tensor_t, tensor, fold_batches, train_gen
from utils import imread_rgb, log, log_detail, get_lab_indicators
from utils import get_images_labels, get_similarities, embeddings_device_dim
from utils import test_print_descriptor, choose_rand_neg, get_pos_couples
from model.siamese import TuneClassifSub, RegionDescriptorNet
from model.custom_modules import TripletLoss

# keep labels as global variable. they are initialized after
# train set has been loaded and then kept constant
labels = []
train_type = P.cnn_model.lower() + ' Siamese sub-regions'


def get_embeddings(net, dataset, device, out_size):
    test_trans = P.test_trans
    if P.test_pre_proc:
        test_trans = transforms.Compose([])

    def batch(last, i, is_final, batch):
        embeddings = last
        # one image at a time
        test_in = move_device(test_trans(batch[0][0]).unsqueeze(0), P.cuda_device)

        out = net(Variable(test_in, volatile=True)).data
        embeddings[i] = out[0]
        return embeddings

    init = tensor(device, len(dataset), out_size)
    return fold_batches(batch, init, dataset, 1)


# train using triplets, constructing triplets from all positive couples
def train_siam_triplets_pos_couples(net, train_set, testset_tuple, criterion, criterion2, optimizer, best_score=0):
    """
        TODO
    """
    train_trans = P.train_trans
    if P.train_pre_proc:
        train_trans = transforms.Compose([])

    couples = get_pos_couples(train_set)
    sim_device, _ = embeddings_device_dim(P, net, len(train_set), sim_matrix=True)
    lab_indicators = get_lab_indicators(train_set, sim_device)
    num_pos = sum(len(couples[lab]) for lab in couples)
    log(P, '#pos (without order, with duplicates):{0}'.format(num_pos))

    # fold over positive couples here and choose negative for each pos
    # need to make sure the couples are evenly distributed
    # such that all batches can have couples from every instance
    def shuffle_couples(couples):
        for lab in couples:
            random.shuffle(couples[lab])
        # get x such that only 20% of labels have more than x couples
        a = np.array([len(couples[lab]) for lab in couples])
        x = int(np.percentile(a, 80))
        out = []
        keys = couples.keys()
        random.shuffle(keys)
        # append the elements to out in a strided way
        # (up to x elements per label)
        for count in range(x):
            for lab in keys:
                if count >= len(couples[lab]):
                    continue
                out.append(couples[lab][count])
        # the last elements in the longer lists are inserted at random
        for lab in keys:
            for i in range(x, len(couples[lab])):
                out.insert(random.randrange(len(out)), couples[lab][i])
        return out

    def create_epoch(epoch, couples, testset_tuple):
        test_ref_set = testset_tuple[1]
        # use the test-train set to obtain embeddings and similarities
        # (since it may be transformed differently than train set)
        similarities, _ = get_similarities(P, get_embeddings, net, test_ref_set)

        # shuffle the couples
        shuffled = shuffle_couples(couples)
        return shuffled, {'epoch': epoch, 'similarities': similarities}

    def create_batch(batch, n, epoch, similarities):
        # one image at a time. batch is always of size 1
        lab, (i1, i2), (im1, im2) = batch[0]
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, 1)
        labels_in[0] = labels.index(lab)
        # we get a positive couple. find negative for it
        im3 = None
        # choose a semi-hard negative. see FaceNet
        # paper by Schroff et al for details.
        # essentially, choose hardest negative that is still
        # easier than the positive. this should avoid
        # collapsing the model at beginning of training
        ind_exl = lab_indicators[lab]
        sim_pos = similarities[i1, i2]
        if epoch < P.train_epoch_switch:
            # exclude all positives as well as any that are
            # more similar than sim_pos
            ind_exl = ind_exl | similarities[i1].ge(sim_pos)
        if ind_exl.sum() >= similarities.size(0):
            p = 'cant find semi-hard neg for'
            s = 'falling back to random neg'
            n_pos = lab_indicators[lab].sum()
            n_ge = similarities[i1].ge(sim_pos).sum()
            n_tot = similarities.size(0)
            print('{0} {1}-{2}-{3} (#pos:{4}, #ge:{5}, #total:{6}), {7}'.format(p, i1, i2, lab, n_pos, n_ge, n_tot, s))
        else:
            # similarities must be in [-1, 1]
            # set all similarities of excluded indexes to -2
            # then take argmax (highest similarity not excluded)
            sims = similarities[i1].clone()
            sims[ind_exl] = -2
            _, k = sims.max(0)
            im3 = train_set[k[0]][0]
        if im3 is None:
            # default to random negative
            im3 = choose_rand_neg(train_set, lab)
        # one image at a time
        train_in1 = move_device(train_trans(im1).unsqueeze(0), P.cuda_device)
        train_in2 = move_device(train_trans(im2).unsqueeze(0), P.cuda_device)
        train_in3 = move_device(train_trans(im3).unsqueeze(0), P.cuda_device)
        # return input tensors and labels
        return [train_in1, train_in2, train_in3], [labels_in]

    def create_loss(out, labels_list):
        # out is a tuple of 3 tuples, each for the descriptor
        # and a tensor with all classification results for the highest
        # classification values. the first loss is a simple loss on the
        # descriptors. the second loss is a classification loss for
        # each sub-region of the anchor input (first input).
        # we simply sum-aggregate here
        loss = criterion(*(t for t, _ in out))
        cls_out = out[0][1]  # classification values for anchor
        # there is only 1 batch of k classification values, so cls_out
        # has dimension (1, num_classes, k). need to get (k, num_classes)
        cls_out_all = cls_out.squeeze(0).t()
        loss2 = criterion2(cls_out_all, labels_list[0].expand(cls_out_all.size(0)))
        return loss, loss2

    train_gen(train_type, P, test_print_descriptor, get_embeddings, net,
              couples, testset_tuple, optimizer, create_epoch, create_batch,
              create_loss, best_score=best_score)


def get_siamese_net():
    model = models.alexnet
    if P.cnn_model.lower() == 'resnet152':
        model = models.resnet152
    class_net = TuneClassifSub(model(pretrained=True), P.num_classes, P.feature_size2d, untrained=P.untrained_blocks)
    if P.classif_model:
        class_net.load_state_dict(torch.load(P.classif_model, map_location=lambda storage, location: storage.cpu()))
    net = RegionDescriptorNet(class_net, P.regions_k, P.feature_dim, P.feature_size2d, untrained=P.untrained_blocks)
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

    siam_net = get_siamese_net()
    optimizer = optim.SGD((p for p in siam_net.parameters() if p.requires_grad), lr=P.train_lr, momentum=P.train_momentum, weight_decay=P.train_weight_decay)
    criterion = TripletLoss(P.triplet_margin, P.train_loss_avg)
    criterion2 = nn.CrossEntropyLoss(size_average=P.train_loss2_avg)
    testset_tuple = (test_set, test_train_set)
    if P.test_upfront:
        log(P, 'Upfront testing of descriptor model')
        score = test_print_descriptor(train_type, P, siam_net, testset_tuple, get_embeddings)
    else:
        score = 0
    if P.train:
        log(P, 'Starting region-descriptor training')
        train_siam_triplets_pos_couples(siam_net, train_set, testset_tuple, criterion, criterion2, optimizer, best_score=score)
        log(P, 'Finished region-descriptor training')
    if P.test_descriptor_net:
        log(P, 'Testing as descriptor')
        # set best score high enough such that it will never be saved
        test_print_descriptor(train_type, P, siam_net, testset_tuple, get_embeddings, best_score=len(test_set) + 1)


if __name__ == '__main__':
    with torch.cuda.device(P.cuda_device):
        try:
            main()
        except:
            log_detail(P, None, traceback.format_exc())
            raise
