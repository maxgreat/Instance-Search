# -*- encoding: utf-8 -*-

import random
import itertools
import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from model.custom_modules import NormalizeL2Fun
from test_params import P

# TODO possibly transform all images before creating couples/triplets
# when using random transformations, this will require a fixed number of
# transformations per image (need to decide on that number)


def get_device_and_size(net, n, sim_matrix=False):
    # get best device for embeddings (and possibly similarity matrix),
    # as well as the feature vector size.
    # usually, this is the configured cuda device.
    # but it could be CPU if embeddings/number of items are too large
    device = P.cuda_device
    out_size = P.siam_feature_dim
    if hasattr(net, 'feature_size'):
        out_size = net.feature_size
    if n * out_size * 4 > P.embeddings_cuda_size:
        # we will consume more than 1 GB here. use CPU
        device = -1
    if sim_matrix and n * n * 4 > P.embeddings_cuda_size:
        device = -1
    return device, out_size


# get all embeddings (feature vectors) of a dataset from a given net
# the net is assumed to be in eval mode
def get_embeddings(net, dataset, device, out_size):
    C, H, W = P.image_input_size
    test_trans = transforms.Compose([])
    if not P.siam_test_pre_proc:
        test_trans.transforms.append(P.siam_test_trans)
    if P.test_norm_per_image:
        test_trans.transforms.append(norm_image_t)

    def batch(last, i, is_final, batch):
        embeddings = last
        n = len(batch)
        test_in = tensor(P.cuda_device, n, C, H, W)
        for j in range(n):
            test_in[j] = test_trans(batch[j][0])

        out = net(Variable(test_in, volatile=True))
        for j in range(n):
            embeddings[i + j] = out.data[j]
        return embeddings
    init = tensor(device, len(dataset), out_size)
    return fold_batches(batch, init, dataset, P.siam_test_batch_size)


# get all similarities between pairs of images of the dataset
# net is assumed to be in train mode
def get_similarities(net, dataset):
    net.eval()
    n = len(dataset)
    d, o = get_device_and_size(net, n, sim_matrix=True)
    embeddings = get_embeddings(net, dataset, d, o)
    similarities = torch.mm(embeddings, embeddings.t())
    net.train()
    return similarities, d


# return a random negative for the given label and train set
def choose_rand_neg(trainSet, lab):
    im_neg, lab_neg = random.choice(trainSet)
    while (lab_neg == lab):
        im_neg, lab_neg = random.choice(trainSet)
    return im_neg


# get byte tensors indicating the indexes of images having a different label
def get_lab_indicators(dataset, device):
    n = len(dataset)
    indicators = {}
    for _, lab1 in dataset:
        if lab1 in indicators:
            continue
        indicator = tensor_t(torch.ByteTensor, device, n).fill_(0)
        for i2, (_, lab2) in enumerate(dataset):
            if lab1 == lab2:
                indicator[i2] = 1
        indicators[lab1] = indicator
    return indicators


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (precision at 1, not the average precision/ranking on the ref set). TODO
def test_descriptor_net(net, testSet, testRefSet, normalized=True):
    d, o = get_device_and_size(net, max(len(testSet), len(testRefSet)))
    test_embeddings = get_embeddings(net, testSet, d, o)
    ref_embeddings = get_embeddings(net, testRefSet, d, o)
    if not normalized:
        test_embeddings = NormalizeL2Fun()(test_embeddings)
        ref_embeddings = NormalizeL2Fun()(ref_embeddings)

    # calculate all similarities as a simple matrix multiplication
    # since inputs are normalized, thus cosine = dot product
    sim = torch.mm(test_embeddings, ref_embeddings.t())
    maxSim, maxIdx = torch.max(sim, 1)
    maxLabel = []
    for i in range(sim.size(0)):
        # get label from ref set which obtained highest score
        maxLabel.append(testRefSet[maxIdx[i, 0]][1])

    # stats
    correct = sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(testSet))
    total = len(testSet)
    sum_pos = sum(sim[i, j] for i, (_, testLabel) in enumerate(testSet) for j, (_, refLabel) in enumerate(testRefSet) if testLabel == refLabel)
    sum_neg = sim.sum() - sum_pos
    sum_max = maxSim.sum()
    lab_dict = dict([(lab, {}) for _, lab in testSet])
    for j, (_, lab) in enumerate(testSet):
        d = lab_dict[lab]
        lab = maxLabel[j]
        d.setdefault(lab, d.get(lab, 0) + 1)
    return correct, total, sum_pos, sum_neg, sum_max, lab_dict


def test_print_siamese(net, testset_tuple, bestScore=0, epoch=0):
    def print_stats(prefix, c, t, avg_pos, avg_neg, avg_max):
        s1 = 'Correct: {0} / {1} -> acc: {2:.4f}\n'.format(c, t, float(c) / t)
        s2 = 'AVG cosine sim (sq dist) values: pos: {0:.4f} ({1:.4f}), neg: {2:.4f} ({3:.4f}), max: {4:.4f} ({5:.4f})'.format(avg_pos, 2 - 2 * avg_pos, avg_neg, 2 - 2 * avg_neg, avg_max, 2 - 2 * avg_max)
        # TODO if not normalized
        P.log(prefix + s1 + s2)

    testSet, testRefSet = testset_tuple
    net.eval()
    correct, tot, sum_pos, sum_neg, sum_max, lab_dict = test_descriptor_net(net, testSet, testRefSet)
    # can save labels dictionary (predicted labels for all test labels)
    # TODO

    num_pos = sum(testLabel == refLabel for _, testLabel in testSet for _, refLabel in testRefSet)
    num_neg = len(testSet) * len(testRefSet) - num_pos

    if (correct > bestScore):
        bestScore = correct
        prefix = 'SIAM, EPOCH:{0}, SCORE:{1}'.format(epoch, correct)
        P.save_uuid(prefix)
        torch.save(net, path.join(P.save_dir, P.unique_str() + "_best_siam.ckpt"))
    print_stats('TEST - ', correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(testSet))
    torch.save(net, path.join(P.save_dir, "model_siam_" + str(epoch) + ".ckpt"))

    # training set accuracy
    trainTestSet = testRefSet[:200]
    correct, tot, sum_pos, sum_neg, sum_max, _ = test_descriptor_net(net, trainTestSet, testRefSet)
    num_pos = sum(testLabel == refLabel for _, testLabel in trainTestSet for _, refLabel in testRefSet)
    num_neg = len(trainTestSet) * len(testRefSet) - num_pos
    print_stats('TRAIN - ', correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(trainTestSet))
    net.train()
    return bestScore


def siam_train_stats(net, testset_tuple, epoch, batchCount, is_last, loss, running_loss, score):
    disp_int = P.siam_loss_int
    test_int = P.siam_test_int
    running_loss += loss
    if batchCount % disp_int == disp_int - 1:
        P.log('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batchCount + 1, running_loss / disp_int))
        running_loss = 0.0
    # test model every x mini-batches
    if ((test_int > 0 and batchCount % test_int == test_int - 1) or
            (test_int <= 0 and is_last)):
        score = test_print_siamese(net, testset_tuple, score, epoch + 1)
    return running_loss, score


def train_siam_couples(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    C, H, W = P.image_input_size
    trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        trans = transforms.Compose([])

    def micro_batch(last, i, is_final, batch):
        n = len(batch)
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_labels = tensor(P.cuda_device, n)
        for j, (i1, (im1, lab)) in enumerate(batch):
            train_in1[j] = trans(im1)
            # with a given probability, choose negative couple,
            # else positive couple
            if random.random() < P.siam_couples_p:
                if P.siam_choice_mode == 'hard':
                    # always choose hardest negative
                    neg_sims = similarities[i1].clone()
                    # similarity is in [-1, 1]
                    neg_sims[lab_indicators[lab]] = -2
                    _, k = neg_sims.max(0)
                    im2 = trainSet[k[0]][0]
                else:
                    # choose random negative for this label
                    im2 = choose_rand_neg(trainSet, lab)
                train_in2[j] = trans(im2)
                train_labels[j] = -1
            else:
                # choose any positive randomly
                train_in2[j] = trans(random.choice(couples[im1]))
                train_labels[j] = 1
        out1, out2 = net(Variable(train_in1), Variable(train_in2))
        loss = criterion(out1, out2, Variable(train_labels))
        loss.backward()
        return last + loss.data[0]

    def train_couples(last, i, is_final, batch):
        batchCount, score, running_loss = last

        # zero the parameter gradients, then forward + back prop
        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, is_final, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    couples = get_pos_couples_ibi(trainSet)
    num_pos = sum(len(couples[im]) for im in couples)
    P.log('#pos (with order, with duplicates):{0}'.format(num_pos))
    idxTrainSet = list(enumerate(trainSet))
    if P.siam_choice_mode == 'hard':
        # get all similarities between embeddings on the test train set
        similarities, sim_device = get_similarities(net, testset_tuple[1])
        lab_indicators = get_lab_indicators(trainSet, sim_device)
    net.train()
    for epoch in range(P.siam_train_epochs):
        optimizer = anneal(net, optimizer, epoch, P.classif_annealing)
        random.shuffle(idxTrainSet)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(train_couples, init, idxTrainSet, P.siam_train_batch_size)
        if P.siam_choice_mode == 'hard':
            # update similarities
            similarities, _ = get_similarities(net, testset_tuple[1])


# train using triplets generated each epoch
def train_siam_triplets(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    C, H, W = P.image_input_size
    train_trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        train_trans = transforms.Compose([])

    n_triplets = len(trainSet)
    if P.siam_choice_mode == 'easy-hard':
        n_triplets *= P.siam_easy_hard_n_t
    P.log('#triplets:{0}'.format(n_triplets))
    sim_device, out_size = get_device_and_size(net, len(trainSet), sim_matrix=True)
    lab_indicators = get_lab_indicators(trainSet, sim_device)
    pos_couples = get_pos_couples_ibi(trainSet)

    def micro_batch(last, i, is_final, batch):
        n = len(batch)
        if P.siam_choice_mode == 'easy-hard':
            n *= P.siam_easy_hard_n_t
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        # we get a batch of triplets
        for j, (im1, im2, im3) in enumerate(batch):
            train_in1[j] = train_trans(im1)
            train_in2[j] = train_trans(im2)
            train_in3[j] = train_trans(im3)
        out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
        loss = criterion(out1, out2, out3)
        loss.backward()
        return last + loss.data[0]

    def train_batch(last, i, is_final, batch):
        batchCount, score, running_loss = last
        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, is_final, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def triplets_rand():
        triplets = []
        for im, lab in trainSet:
            im_pos = random.choice(pos_couples[im])
            im_neg = choose_rand_neg(trainSet, lab)
            triplets.append((im, im_pos, im_neg))
        return triplets

    def triplets_easy_hard(similarities, embeddings):
        triplets = []
        for i_im, (im, lab) in enumerate(trainSet):
            ind_pos = lab_indicators[lab]
            ind_neg = t_not(ind_pos)
            n_n = int(min(P.siam_easy_hard_n_n, ind_neg.sum()))
            n_p = int(min(P.siam_easy_hard_n_p, ind_pos.sum()))

            sim_negs = similarities[i_im].clone()
            # similarity is in [-1, 1]
            sim_negs[ind_pos] = -2
            _, hard_negs = sim_negs.topk(n_n)

            sim_pos = similarities[i_im].clone()
            sim_pos[ind_neg] = -2
            _, easy_pos = sim_pos.topk(n_p)
            # get the loss values and keep the highest n_t ones
            # we calculate loss as x_a (x_n - x_p) for each triplet
            # since they are normalized and we can ignore margin here
            n_t = P.siam_easy_hard_n_t
            # get negative and positive embeddings as matrix
            neg_embs = embeddings[hard_negs]
            pos_embs = embeddings[easy_pos]
            # repeat the matrices to obtain all possible combinations
            # of x_n - x_p
            neg_embs = neg_embs.repeat(n_p, 1)
            pos_embs = pos_embs.repeat(1, n_n).view(n_p * n_n, embeddings.size(1))
            neg_embs.add_(-1, pos_embs)
            anchor_embs = embeddings[i_im].view(1, -1).expand_as(neg_embs)
            loss_values = anchor_embs.mul(neg_embs).sum(1).view(-1)
            _, top_idx = loss_values.topk(n_t)
            for k, idx in enumerate(top_idx):
                i_neg, i_pos = idx % n_n, idx // n_n
                triplets.append((im, trainSet[i_pos][0], trainSet[i_neg][0]))
        return triplets

    net.train()
    triplets = triplets_rand()
    for epoch in range(P.siam_train_epochs):
        optimizer = anneal(net, optimizer, epoch, P.classif_annealing)
        if P.siam_choice_mode == 'easy-hard':
            # in each epoch, update embeddings/similarities
            # get the values from the test-train set
            net.eval()
            embeddings = get_embeddings(net, testset_tuple[1], sim_device, out_size)
            similarities = torch.mm(embeddings, embeddings.t())
            triplets = triplets_easy_hard(similarities, embeddings)
            net.train()
        random.shuffle(triplets)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(train_batch, init, triplets, P.siam_train_batch_size)


# train using triplets, constructing triplets from all positive couples
def train_siam_triplets_pos_couples(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    C, H, W = P.image_input_size
    train_trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        train_trans = transforms.Compose([])

    def micro_batch(last, i, is_final, batch):
        n = len(batch)
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        # we get a batch of positive couples
        # find negatives for each couple
        for j, (lab, (i1, i2), (im1, im2)) in enumerate(batch):
            im3 = None
            if P.siam_choice_mode == 'hard':
                # choose hardest negative every time
                ind_exl = lab_indicators[lab]
                sims = similarities[i1].clone()
                # similarity is in [-1, 1]
                sims[ind_exl] = -2
                _, k = sims.max(0)
                im3 = trainSet[k[0]][0]
            elif P.siam_choice_mode == 'semi-hard':
                # choose a semi-hard negative. see FaceNet
                # paper by Schroff et al for details.
                # essentially, choose hardest negative that is still
                # easier than the positive. this should avoid
                # collapsing the model at beginning of training
                ind_exl = lab_indicators[lab]
                sim_pos = similarities[i1, i2]
                if epoch < P.siam_triplets_switch:
                    # exclude all positives as well as any that are
                    # more similar than sim_pos
                    ind_exl = ind_exl | similarities[i1].ge(sim_pos)
                if ind_exl.sum() >= similarities.size(0):
                    p = 'cant find semi-hard neg for'
                    s = 'falling back to random neg'
                    P.log('{0} {1}-{2}-{3}, {4}'.format(p, i1, i2, lab, s))
                else:
                    sims = similarities[i1].clone()
                    sims[ind_exl] = -2
                    _, k = sims.max(0)
                    im3 = trainSet[k[0]][0]
            if im3 is None:
                # default to random negative
                im3 = choose_rand_neg(trainSet, lab)
            train_in1[j] = train_trans(im1)
            train_in2[j] = train_trans(im2)
            train_in3[j] = train_trans(im3)
        out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
        loss = criterion(out1, out2, out3)
        loss.backward()
        return last + loss.data[0]

    def train_triplets(last, i, is_final, batch):
        batchCount, score, running_loss = last

        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, is_final, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def shuffle_couples(couples):
        for l in couples:
            random.shuffle(couples[l])
        # get x such that only 20% of labels have more than x couples
        a = np.array([len(couples[l]) for l in couples])
        x = int(np.percentile(a, 80))
        out = []
        keys = couples.keys()
        random.shuffle(keys)
        # append the elements to out in a strided way
        # (up to x elements per label)
        for count in range(x):
            for l in keys:
                if count >= len(couples[l]):
                    continue
                out.append(couples[l][count])
        # the last elements in the longer lists are inserted at random
        for l in keys:
            for i in range(x, len(couples[l])):
                out.insert(random.randrange(len(out)), couples[l][i])
        return out

    couples = get_pos_couples(trainSet)
    num_train = num_pos = sum(len(couples[l]) for l in couples)
    P.log('#pos (without order, with duplicates):{0}'.format(num_pos))

    sim_device, _ = get_device_and_size(net, len(trainSet), sim_matrix=True)
    lab_indicators = get_lab_indicators(trainSet, sim_device)
    net.train()
    for epoch in range(P.siam_train_epochs):
        optimizer = anneal(net, optimizer, epoch, P.classif_annealing)
        # use the test-train set to obtain embeddings and similarities
        # (since it may be transformed differently than train set)
        similarities, sim_device = get_similarities(net, testset_tuple[1])

        # fold over positive couples here and choose negative for each pos
        # need to make sure the couples are evenly distributed
        # such that all batches can have couples from every instance
        shuffled = shuffle_couples(couples)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(train_triplets, init, shuffled, P.siam_train_batch_size)
