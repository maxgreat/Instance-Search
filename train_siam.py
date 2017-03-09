# -*- encoding: utf-8 -*-

from torch.autograd import Variable

from os import path

from utils import *
from model.siamese import Normalize2DL2
from test_params import P

# TODO create generator to yield couples of images
# / triplets (need a way to identify positive couples for each images,
# then iterate over all others to create triples)


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (not the average precision/ranking on the ref set). TODO
def test_descriptor_net(net, testSet, testRefSet, normalized=True):
    normalize_rows = Normalize2DL2()

    def eval_batch_ref(last, i, batch):
        maxSim, maxLabel, sum_pos, sum_neg, out1, testLabels = last
        C, H, W = P.siam_input_size
        test_in2 = tensor(P.cuda_device, len(batch), C, H, W)
        for k, (refIm, _) in enumerate(batch):
            if P.siam_test_pre_proc:
                test_in2[k] = refIm
            else:
                test_in2[k] = P.siam_test_trans(refIm)
        out2 = net(Variable(test_in2, volatile=True)).data
        if not normalized:
            out2 = normalize_rows(out2)
        sim = torch.mm(out1, out2.t())
        sum_pos += sum(sim[j, k] for j, testLabel in enumerate(testLabels) for k, (_, refLabel) in enumerate(batch) if testLabel == refLabel)
        sum_neg += (sim.sum() - sum_pos)
        batchMaxSim, batchMaxIdx = torch.max(sim, 1)
        for j in range(maxSim.size(0)):
            if (batchMaxSim[j, 0] > maxSim[j, 0]):
                maxSim[j, 0] = batchMaxSim[j, 0]
                maxLabel[j] = batch[batchMaxIdx[j, 0]][1]
        return maxSim, maxLabel, sum_pos, sum_neg, out1, testLabels

    def eval_batch_test(last, i, batch):
        correct, total, sum_pos, sum_neg, sum_max, lab_dict = last
        C, H, W = P.siam_input_size
        test_in1 = tensor(P.cuda_device, len(batch), C, H, W)
        for j, (testIm, _) in enumerate(batch):
            if P.siam_test_pre_proc:
                test_in1[j] = testIm
            else:
                test_in1[j] = P.siam_test_trans(testIm)
        out1 = net(Variable(test_in1, volatile=True)).data
        if not normalized:
            out1 = normalize_rows(out1)
        # max similarity, max label, outputs
        maxSim = tensor(P.cuda_device, len(batch), 1).fill_(-2)
        init = maxSim, [None for _ in batch], sum_pos, sum_neg, out1, [lab for im, lab in batch]
        maxSim, maxLabel, sum_pos, sum_neg, _, _ = fold_batches(eval_batch_ref, init, testRefSet, P.siam_test_batch_size)
        sum_max += maxSim.sum()
        for j, (_, lab) in enumerate(batch):
            lab_dict[lab].append((maxLabel[j], 1))
        total += len(batch)
        correct += sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(batch))
        return correct, total, sum_pos, sum_neg, sum_max, lab_dict

    lab_dict = dict([(lab, []) for _, lab in testSet])
    return fold_batches(eval_batch_test, (0, 0, 0.0, 0.0, 0.0, lab_dict), testSet, P.siam_test_batch_size)


def test_print_siamese(net, testset_tuple, bestScore=0, epoch=0):
    def print_stats(prefix, c, t, avg_pos, avg_neg, avg_max):
        s1 = 'Correct: {0} / {1} -> acc: {2:.4f}\n'.format(c, t, float(c) / t)
        s2 = 'AVG cosine sim values: pos: {0:.4f}, neg: {1:.4f}, max: {2:.4f}\n'.format(avg_pos, avg_neg, avg_max)
        # TODO if not normalized
        avg_pos = 2 - 2 * avg_pos
        avg_neg = 2 - 2 * avg_neg
        avg_max = 2 - 2 * avg_max
        s3 = 'AVG squared dist values: pos: {0:.4f}, neg: {1:.4f}, max: {2:.4f}'.format(avg_pos, avg_neg, avg_max)
        print(prefix + s1 + s2 + s3)

    testSet, testRefSet = testset_tuple
    net.eval()
    correct, tot, sum_pos, sum_neg, sum_max, lab_dict = test_descriptor_net(net, testSet, testRefSet)
    # can save labels dictionary (predicted labels for all test labels)
    # for lab in lab_dict:
    #     def red_f(x, y):
    #         return x[0], x[1] + y[1]
    #     L = sorted(lab_dict[lab], key=lambda x: x[0])
    #     g = itertools.groupby(L, key=lambda x: x[0])
    #     red = [reduce(red_f, group) for _, group in g]
    #     lab_dict[lab] = sorted(red, key=lambda x: -x[1])
    # f = open(saveDir + 'lab_dict_' + str(epoch) + '.txt', 'w')
    # for lab in lab_dict:
    #     f.write(str(lab) + ':' + str(lab_dict[lab]) + '\n')
    # f.close()

    num_pos = sum(testLabel == refLabel for _, testLabel in testSet for _, refLabel in testRefSet)
    num_neg = len(testSet) * len(testRefSet) - num_pos

    if (correct > bestScore):
        bestScore = correct
        prefix = 'SIAM, EPOCH:{0}, SCORE:{1}'.format(epoch, correct)
        P.save_uuid(prefix)
        torch.save(net, path.join(P.save_dir, P.uuid.hex + "_best_siam.ckpt"))
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


def siam_train_stats(net, testset_tuple, epoch, batchCount, loss, running_loss, score):
    disp_int = P.siam_loss_int
    test_int = P.siam_test_int
    running_loss += loss.data[0]
    if batchCount % disp_int == disp_int - 1:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batchCount + 1, running_loss / disp_int))
        running_loss = 0.0
    # test model every x mini-batches
    if batchCount % test_int == test_int - 1:
        score = test_print_siamese(net, testset_tuple, score, epoch + 1)
    return running_loss, score


def train_siam_couples(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    def train_couples(last, i, batch):
        batchCount, score, running_loss = last

        # using sub-batches (only pairs with biggest loss)
        # losses = []
        # TODO

        # get the inputs
        n = len(batch)
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_labels = tensor(P.cuda_device, n)
        for j, ((im1, im2), lab) in enumerate(batch):
            if P.siam_train_pre_proc:
                train_in1[j] = im1
                train_in2[j] = im2
            else:
                train_in1[j] = trans(im1)
                train_in2[j] = trans(im2)
            train_labels[j] = lab

        # zero the parameter gradients, then forward + back prop
        optimizer.zero_grad()
        out1, out2 = net(Variable(train_in1), Variable(train_in2))
        loss = criterion(out1, out2, Variable(train_labels))
        loss.backward()
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def label_f(i1, l1, i2, l2):
        return 1 if l1 == l2 else -1
    couples = get_couples(trainSet, P.siam_couples_p, label_f)
    num_train = len(couples)
    num_pos = sum(1 for _, lab in couples if lab == 1)
    print('training set size:', num_train, '#pos:', num_pos, '#neg:', num_train - num_pos)
    for epoch in range(P.siam_train_epochs):
        random.shuffle(couples)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(f, init, couples, P.siam_train_batch_size)


def train_siam_triplets(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    C, H, W = P.siam_input_size
    trans = P.siam_train_trans

    def embeddings_batch(last, i, batch):
        embeddings = last
        n = len(batch)
        test_in = tensor(P.cuda_device, n, C, H, W)
        for j in range(n):
            if P.siam_test_pre_proc:
                test_in[j] = batch[j][0]
            else:
                test_in[j] = P.siam_test_trans(batch[j][0])
        out = net(Variable(test_in, volatile=True))
        for j in range(n):
            embeddings[i + j] = out.data[j]
        return embeddings

    def train_triplets(last, i, batch):
        batchCount, score, running_loss = last
        # we get a batch of positive couples
        # find random negatives for each couple
        n = len(batch)
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        for j, (lab, _, (x1, x2)) in enumerate(batch):
            k = random.randrange(len(trainSet))
            while (trainSet[k][1] == lab):
                k = random.randrange(len(trainSet))
            if P.siam_train_pre_proc:
                train_in1[j] = x1
                train_in2[j] = x2
                train_in3[j] = trainSet[k][0]
            else:
                train_in1[j] = trans(x1)
                train_in2[j] = trans(x2)
                train_in3[j] = trans(trainSet[k][0])

        optimizer.zero_grad()
        out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
        loss = criterion(out1, out2, out3)
        loss.backward()
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def train_triplets_hard(last, i, batch):
        batchCount, score, running_loss = last
        # we get a batch of positive couples
        # for each couple, find a negative such that the embedding is
        # semi-hard (using the one with smallest distance can collapse
        # the model, according to Schroff et al - FaceNet) so find
        # one that lies in the margin (alpha) used by the loss to
        # discriminate between positive and negative pair
        # for normalized vectors x and y, we have ||x-y||^2 = 2 - 2xy
        # so finding a negative example such that ||x-x_p||^2 < ||x-x_n||^2
        # is equivalent to having x.x_p > x.x_n
        # after x epochs, we only take the hardest negative examples
        n = len(batch)
        train_in1 = tensor(P.cuda_device, n, C, H, W)
        train_in2 = tensor(P.cuda_device, n, C, H, W)
        train_in3 = tensor(P.cuda_device, n, C, H, W)
        for j, (lab, (i1, i2), (x1, x2)) in enumerate(batch):
            em1 = embeddings[i1]
            sqdist_pos = (em1 - embeddings[i2]).pow(2).sum()
            negatives = []
            for k, embedding in enumerate(embeddings):
                if trainSet[k][1] == lab:
                    continue
                sqdist_neg = (em1 - embedding).pow(2).sum()
                if epoch < P.siam_triplets_switch and sqdist_pos >= sqdist_neg:
                    continue
                negatives.append((k, sqdist_neg))
            if len(negatives) <= 0:
                # print('cannot find a semi-hard negative for {0}-{1}-{2}. falling back to random negative'.format(i1, i2, lab))
                k = random.randrange(len(trainSet))
                while (trainSet[k][1] == lab):
                    k = random.randrange(len(trainSet))
                x3 = trainSet[k][0]
            else:
                k3 = min(negatives, key=lambda x: x[1])[0]
                x3 = trainSet[k3][0]
            if P.siam_train_pre_proc:
                train_in1[j] = x1
                train_in2[j] = x2
                train_in3[j] = x3
            else:
                train_in1[j] = trans(x1)
                train_in2[j] = trans(x2)
                train_in3[j] = trans(x3)

        optimizer.zero_grad()
        out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
        loss = criterion(out1, out2, out3)
        loss.backward()
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, loss, running_loss, score)
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

    # for triplets, only fold over positive couples.
    # then choose negative for each couple specifically
    couples = get_pos_couples(trainSet)
    num_pos = sum(len(couples[l]) for l in couples)
    print('#pos:{0}'.format(num_pos))
    if P.siam_train_mode == 'triplets':
        f = train_triplets
    else:
        f = train_triplets_hard

    for epoch in range(P.siam_train_epochs):
        # for the 'hard' triplets, we need to know the embeddings of all
        # images at each epoch. so pre-calculate them here
        if P.siam_train_mode == 'triplets_hard':
            init = tensor(P.cuda_device, len(trainSet), P.siam_feature_dim)
            net.eval()
            # use the test-train set to obtain embeddings
            # (since it may be transformed differently than train set)
            embeddings = fold_batches(embeddings_batch, init, testset_tuple[1], P.siam_test_batch_size)
            net.train()

        # for triplets, need to make sure the couples are evenly
        # distributed (such that all batches can have couples from
        # every instance)
        shuffled = shuffle_couples(couples)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(f, init, shuffled, P.siam_train_batch_size)
