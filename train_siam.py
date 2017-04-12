# -*- encoding: utf-8 -*-

import random
import math
import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from model.siamese import Siamese2
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
        device = -1
    if sim_matrix and n * n * 4 > P.embeddings_cuda_size:
        device = -1
    return device, out_size


# get all embeddings (feature vectors) of a dataset from a given net
# the net is assumed to be in eval mode
def get_embeddings(net, dataset, device, out_size):
    is_siam2 = isinstance(net, Siamese2)
    C, H, W = P.image_input_size
    test_trans = transforms.Compose([])
    if not P.siam_test_pre_proc:
        test_trans.transforms.append(P.siam_test_trans)
    if P.test_norm_per_image:
        test_trans.transforms.append(norm_image_t)

    def batch(last, i, is_final, batch):
        embeddings = last
        n = len(batch)
        if is_siam2:
            # one image at a time
            test_in = move_device(batch[0][0].unsqueeze(0), P.cuda_device)
        else:
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
def choose_rand_neg(train_set, lab):
    im_neg, lab_neg = random.choice(train_set)
    while (lab_neg == lab):
        im_neg, lab_neg = random.choice(train_set)
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
def test_descriptor_net(net, test_set, test_ref_set, kth=1, normalized=True):
    d, o = get_device_and_size(net, max(len(test_set), len(test_ref_set)))
    test_embeddings = get_embeddings(net, test_set, d, o)
    ref_embeddings = get_embeddings(net, test_ref_set, d, o)
    if not normalized:
        test_embeddings = NormalizeL2Fun()(test_embeddings)
        ref_embeddings = NormalizeL2Fun()(ref_embeddings)

    # calculate all similarities as a simple matrix multiplication
    # since inputs are normalized, thus cosine = dot product
    # the similarities here should always be on CPU
    # (kthvalue is only implemented there and we don't need GPU perf)
    sim = torch.mm(test_embeddings, ref_embeddings.t()).cpu()
    # stats
    prec1, correct, total, max_sim, max_label = precision1(sim, test_set, test_ref_set, kth)
    mAP = mean_avg_precision(sim, test_set, test_ref_set, kth)
    sum_pos = sum(sim[i, j] for i, (_, test_label) in enumerate(test_set) for j, (_, ref_label) in enumerate(test_ref_set) if test_label == ref_label)
    sum_neg = sim.sum() - sum_pos
    sum_max = max_sim.sum()
    lab_dict = dict([(lab, {}) for _, lab in test_set])
    for j, (_, lab) in enumerate(test_set):
        d = lab_dict[lab]
        lab = max_label[j]
        d.setdefault(lab, d.get(lab, 0) + 1)
    return prec1, correct, total, sum_pos, sum_neg, sum_max, mAP, lab_dict


def test_print_siamese(net, testset_tuple, best_score=0, epoch=0):
    def print_stats(prefix, p1, c, t, avg_pos, avg_neg, avg_max, mAP):
        s1 = 'Correct: {0} / {1} - acc: {2:.4f} - mAP:{3:.4f}\n'.format(c, t, p1, mAP)
        s2 = 'AVG cosine sim (sq dist) values: pos: {0:.4f} ({1:.4f}), neg: {2:.4f} ({3:.4f}), max: {4:.4f} ({5:.4f})'.format(avg_pos, 2 - 2 * avg_pos, avg_neg, 2 - 2 * avg_neg, avg_max, 2 - 2 * avg_max)
        # TODO if not normalized
        P.log(prefix + s1 + s2)

    test_set, test_ref_set = testset_tuple
    net.eval()
    prec1, correct, tot, sum_pos, sum_neg, sum_max, mAP, lab_dict = test_descriptor_net(net, test_set, test_ref_set)
    # can save labels dictionary (predicted labels for all test labels)
    # TODO

    num_pos = sum(test_label == ref_label for _, test_label in test_set for _, ref_label in test_ref_set)
    num_neg = len(test_set) * len(test_ref_set) - num_pos

    if (correct > best_score):
        best_score = correct
        prefix = 'SIAM, EPOCH:{0}, SCORE:{1}'.format(epoch, correct)
        P.save_uuid(prefix)
        torch.save(net.state_dict(), path.join(P.save_dir, P.unique_str() + "_best_siam.pth.tar"))
    print_stats('TEST - ', prec1, correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(test_set), mAP)
    torch.save(net.state_dict(), path.join(P.save_dir, "model_siam_" + str(epoch) + ".pth.tar"))

    # training set accuracy (choose second highest value,
    # as highest should almost certainly be the same image)
    # choose train samples with at least 2 other images for the query
    couples = get_pos_couples(test_ref_set)
    train_test_set = random.sample(test_ref_set, 200)
    train_test_set = filter(lambda x: len(couples[x[1]]) >= 3, train_test_set)
    prec1, correct, tot, sum_pos, sum_neg, sum_max, mAP, _ = test_descriptor_net(net, train_test_set, test_ref_set, kth=2)
    num_pos = sum(test_label == ref_label for _, test_label in train_test_set for _, ref_label in test_ref_set)
    num_neg = len(train_test_set) * len(test_ref_set) - num_pos
    print_stats('TRAIN - ', prec1, correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(train_test_set), mAP)
    net.train()
    return best_score


def siam_train_stats(net, testset_tuple, epoch, batch_count, is_last, loss, running_loss, score):
    disp_int = P.siam_loss_int
    test_int = P.siam_test_int
    running_loss += loss
    if batch_count % disp_int == disp_int - 1:
        P.log('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batch_count + 1, running_loss / disp_int))
        running_loss = 0.0
    # test model every x mini-batches
    if ((test_int > 0 and batch_count % test_int == test_int - 1) or
            (test_int <= 0 and is_last)):
        score = test_print_siamese(net, testset_tuple, score, epoch + 1)
    return running_loss, score


def train_siam_couples(net, train_set, testset_tuple, labels, criterion, optimizer, criterion2=None, best_score=0):
    C, H, W = P.image_input_size
    trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        trans = transforms.Compose([])

    couples = get_pos_couples_ibi(train_set)
    num_pos = sum(len(couples[im]) for im in couples)
    P.log('#pos (with order, with duplicates):{0}'.format(num_pos))
    idx_train_set = list(enumerate(train_set))
    sim_device, _ = get_device_and_size(net, len(train_set), sim_matrix=True)
    lab_indicators = get_lab_indicators(train_set, sim_device)

    def create_epoch(epoch, idx_train_set, testset_tuple):
        test_set, test_ref_set = testset_tuple
        similarities, _ = get_similarities(net, test_ref_set)
        random.shuffle(idx_train_set)
        return idx_train_set, {'similarities': similarities}, {}

    def create_batch(batch, n, similarities):
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_labels = tensor(P.cuda_device, n)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n)
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
                    im2 = train_set[k[0]][0]
                else:
                    # choose random negative for this label
                    im2 = choose_rand_neg(train_set, lab)
                train_in2[j] = trans(im2)
                train_labels[j] = -1
            else:
                # choose any positive randomly
                train_in2[j] = trans(random.choice(couples[im1]))
                train_labels[j] = 1
            labels_in[j] = labels.index(lab)
        return [train_in1, train_in2], [train_labels, labels_in]

    def create_loss(tensors_out, labels_in):
        # couple of output descriptors and the train labels
        loss = criterion(tensors_out[0], tensors_out[1], labels_in[0])
        loss2 = None
        if criterion2:
            loss2 = criterion2(tensors_out[0], labels_in[1])
        return loss, loss2

    train_gen(False, net, idx_train_set, testset_tuple, optimizer, P, create_epoch, create_batch, siam_train_stats, create_loss, best_score=best_score)


# train using triplets generated each epoch
def train_siam_triplets(net, train_set, testset_tuple, labels, criterion, optimizer, criterion2=None, best_score=0):
    C, H, W = P.image_input_size
    train_trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        train_trans = transforms.Compose([])

    sim_device, out_size = get_device_and_size(net, len(train_set), sim_matrix=True)
    lab_indicators = get_lab_indicators(train_set, sim_device)
    pos_couples = get_pos_couples_ibi(train_set)

    def triplets_rand():
        triplets = []
        for im, lab in train_set:
            im_pos = random.choice(pos_couples[im])
            im_neg = choose_rand_neg(train_set, lab)
            triplets.append((lab, im, im_pos, im_neg))
        return triplets

    def triplets_easy_hard(train_set, similarities, embeddings):
        triplets = []
        for i_im, (im, lab) in enumerate(train_set):
            ind_pos = lab_indicators[lab]
            ind_neg = t_not(ind_pos)
            # to avoid duplicates pos pairs, consider all images
            # before this one as pos and neg, to exclude them from all
            n_p = int(min(P.siam_eh_n_p, ind_pos[i_im:].sum()))
            n_n = int(min(P.siam_eh_n_n, ind_neg.sum()))
            if i_im > 0:
                ind_neg[:i_im] = 1
            sim_pos = similarities[i_im].clone()
            sim_pos[ind_neg] = -2
            _, easy_pos = sim_pos.topk(n_p)
            sim_negs = similarities[i_im].clone()
            # similarity is in [-1, 1]
            sim_negs[ind_pos] = -2
            _, hard_negs = sim_negs.topk(n_n)

            # get the loss values and keep the highest n_t ones
            # we calculate loss as x_a (x_n - x_p) for each triplet
            # since they are normalized and we can ignore margin here
            n_t = min(P.siam_eh_n_t, n_p, n_n)
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
                triplets.append((lab, im, train_set[i_pos][0], train_set[i_neg][0]))
        # sample only as many triplets as required
        if P.siam_eh_req_triplets >= len(triplets):
            random.shuffle(triplets)
            return triplets
        return random.sample(triplets, P.siam_eh_req_triplets)

    def create_epoch(epoch, train_set, test_set):
        if P.siam_choice_mode == 'easy-hard':
            test_set, test_ref_set = test_set
            net.eval()
            embeddings = get_embeddings(net, test_ref_set, sim_device, out_size)
            similarities = torch.mm(embeddings, embeddings.t())
            triplets = triplets_easy_hard(train_set, similarities, embeddings)
            net.train()
        else:
            triplets = triplets_rand()
        if epoch <= 0:
            P.log('#triplets:{0}'.format(len(triplets)))
        return triplets, {}, {}

    def create_batch(batch, n):
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n)
        for j, (lab, im1, im2, im3) in enumerate(batch):
            train_in1[j] = train_trans(im1)
            train_in2[j] = train_trans(im2)
            train_in3[j] = train_trans(im3)
            labels_in[j] = labels.index(lab)
        return [train_in1, train_in2, train_in3], [labels_in]

    def create_loss(tensors_out, labels_in):
        loss = criterion(*tensors_out)
        loss2 = None
        if criterion2:
            loss2 = criterion2(tensors_out[0], labels_in[0])
        return loss, loss2

    train_gen(False, net, train_set, test_set, optimizer, P, create_epoch, create_batch, siam_train_stats, create_loss, best_score=best_score)


# train using triplets, constructing triplets from all positive couples
# if Siamese2 network, use adapted loss definition
def train_siam_triplets_pos_couples(net, train_set, testset_tuple, labels, criterion, optimizer, criterion2=None, best_score=0):
    """
        TODO
    """
    is_siam2 = isinstance(net, Siamese2)
    C, H, W = P.image_input_size
    train_trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        train_trans = transforms.Compose([])

    couples = get_pos_couples(train_set)
    sim_device, _ = get_device_and_size(net, len(train_set), sim_matrix=True)
    lab_indicators = get_lab_indicators(train_set, sim_device)
    num_pos = sum(len(couples[l]) for l in couples)
    P.log('#pos (without order, with duplicates):{0}'.format(num_pos))

    # fold over positive couples here and choose negative for each pos
    # need to make sure the couples are evenly distributed
    # such that all batches can have couples from every instance
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

    def create_epoch(epoch, couples, test_set):
        test_set, test_ref_set = test_set
        # use the test-train set to obtain embeddings and similarities
        # (since it may be transformed differently than train set)
        similarities, _ = get_similarities(net, test_ref_set)

        # shuffle the couples
        shuffled = shuffle_couples(couples)
        return shuffled, {'epoch': epoch, 'similarities': similarities}, {}

    def create_batch(batch, n, epoch, similarities):
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n)
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
                im3 = train_set[k[0]][0]
            elif P.siam_choice_mode == 'semi-hard':
                # choose a semi-hard negative. see FaceNet
                # paper by Schroff et al for details.
                # essentially, choose hardest negative that is still
                # easier than the positive. this should avoid
                # collapsing the model at beginning of training
                ind_exl = lab_indicators[lab]
                sim_pos = similarities[i1, i2]
                if epoch < P.siam_sh_epoch_switch:
                    # exclude all positives as well as any that are
                    # more similar than sim_pos
                    ind_exl = ind_exl | similarities[i1].ge(sim_pos)
                if ind_exl.sum() >= similarities.size(0):
                    p = 'cant find semi-hard neg for'
                    s = 'falling back to random neg'
                    n_pos = lab_indicators[lab].sum()
                    n_ge = similarities[i1].ge(sim_pos).sum()
                    n_tot = similarities.size(0)
                    P.log('{0} {1}-{2}-{3} (#pos:{4}, #ge:{5}, #total:{6}), {7}'.format(p, i1, i2, lab, n_pos, n_ge, n_tot, s))
                else:
                    sims = similarities[i1].clone()
                    sims[ind_exl] = -2
                    _, k = sims.max(0)
                    im3 = train_set[k[0]][0]
            if im3 is None:
                # default to random negative
                im3 = choose_rand_neg(train_set, lab)
            if is_siam2:
                # one image at a time
                train_in1 = move_device(train_trans(im1).unsqueeze(0), P.cuda_device)
                train_in2 = move_device(train_trans(im2).unsqueeze(0), P.cuda_device)
                train_in3 = move_device(train_trans(im3).unsqueeze(0), P.cuda_device)
            else:
                train_in1[j] = train_trans(im1)
                train_in2[j] = train_trans(im2)
                train_in3[j] = train_trans(im3)
            labels_in[j] = labels.index(lab)
        # return input tensors, no labels
        return [train_in1, train_in2, train_in3], [labels_in]

    if is_siam2:
        def create_loss(out, labels_in):
            # out is a tuple of 3 tuples, each for the descriptor
            # and a tensor with all classification results for the highest
            # classification values. the first loss is a simple loss on the
            # descriptors. the second loss is a classification loss for
            # each sub-region of the input. we simply sum-aggregate here
            loss = criterion(*(t for t, _ in out))
            cls_out = out[0][1]
            loss2 = criterion2(cls_out[:, :, 0], labels_in[0])
            k = cls_out.size(2)
            for i in range(1, k):
                loss2 += criterion2(cls_out[:, :, i], labels_in[0])
            if P.siam_do_loss2_avg:
                loss2 /= k
            return loss, loss2
    else:
        def create_loss(tensors_out, labels_in):
            loss = criterion(*tensors_out)
            loss2 = None
            if criterion2:
                loss2 = criterion2(tensors_out[0], labels_in[0])
            return loss, loss2

    train_gen(False, net, couples, testset_tuple, optimizer, P, create_epoch, create_batch, siam_train_stats, create_loss, best_score=best_score)
