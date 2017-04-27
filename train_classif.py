# -*- encoding: utf-8 -*-

import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from model.nn_utils import set_net_train
from test_params import P


# test a classifier model. it should be in eval mode
def test_classif_net(net, test_set, labels, batchSize):
    """
        Test the network accuracy on a test_set
        Return the number of succes and the number of evaluations done
    """
    C, H, W = P.image_input_size
    trans = transforms.Compose([])
    if not P.classif_test_pre_proc:
        trans.transforms.append(P.classif_test_trans)
    if P.test_norm_per_image:
        trans.transforms.append(norm_image_t)

    def eval_batch_test(last, i, is_final, batch, net, labels):
        correct, total = last
        if P.classif_train_mode == 'subparts':
            # take highest activation of subparts, can only be done with
            # batch size 1
            test_in = move_device(batch[0][0].unsqueeze(0), P.cuda_device)
        else:
            test_in = tensor(P.cuda_device, len(batch), C, H, W)
            for j, (testIm, _) in enumerate(batch):
                test_in[j] = trans(testIm)

        out = net(Variable(test_in, volatile=True)).data
        if P.classif_train_mode == 'subparts':
            max_pred, predicted = torch.max(out, 1)
            _, max_subp = torch.max(max_pred.view(-1), 0)
            predicted = predicted.view(-1)[max_subp[0]]
            total += 1
            correct += (labels.index(batch[0][1]) == predicted)
        else:
            _, predicted = torch.max(out, 1)
            total += len(batch)
            correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), test_set, batchSize, add_args={'net': net, 'labels': labels})


def test_print_classif(net, testset_tuple, labels, best_score=0, epoch=0):
    test_set, test_train_set = testset_tuple
    set_net_train(net, False)
    c, t = test_classif_net(net, test_set, labels, P.classif_test_batch_size)
    if (c > best_score):
        best_score = c
        prefix = 'CLASSIF, EPOCH:{0}, SCORE:{1}'.format(epoch, c)
        P.save_uuid(prefix)
        torch.save(net.state_dict(), path.join(P.save_dir, P.unique_str() + "_best_classif.pth.tar"))
    P.log('TEST - correct: {0} / {1} - acc: {2}'.format(c, t, float(c) / t))

    c, t = test_classif_net(net, test_train_set, labels, P.classif_test_batch_size)
    torch.save(net.state_dict(), path.join(P.save_dir, "model_classif_" + str(epoch) + ".pth.tar"))
    P.log("TRAIN - correct: {0} / {1} - acc: {2}".format(c, t, float(c) / t))
    set_net_train(net, True, bn_train=P.classif_train_bn)
    return best_score


def output_stats(net, testset_tuple, epoch, batch_count, is_final, loss, running_loss, score, labels):
    disp_int = P.classif_loss_int
    running_loss += loss
    if batch_count % disp_int == disp_int - 1:
        P.log('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batch_count + 1, running_loss / disp_int))
        running_loss = 0.0
    test_int = P.classif_test_int
    if ((test_int > 0 and batch_count % test_int == test_int - 1) or
            (test_int <= 0 and is_final)):
        score = test_print_classif(net, testset_tuple, labels, score, epoch + 1)
    return running_loss, score


def train_classif(net, train_set, testset_tuple, labels, criterion, optimizer, best_score=0):
    """
        TODO
    """
    C, H, W = P.image_input_size
    trans = P.classif_train_trans
    if P.classif_train_pre_proc:
        trans = transforms.Compose([])

    def create_epoch(epoch, train_set, testset_tuple):
        random.shuffle(train_set)
        # labels are needed for stats
        return train_set, {}, {'labels': labels}

    def create_batch(batch, n):
        train_in = tensor(P.cuda_device, n, C, H, W)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n)
        for j, (im, lab) in enumerate(batch):
            train_in[j] = trans(im)
            labels_in[j] = labels.index(lab)
        return [train_in], [labels_in]

    def create_loss(t_out, labels_in):
        return criterion(t_out, labels_in[0]), None  # no double loss

    train_gen(True, net, train_set, testset_tuple, optimizer, P, create_epoch, create_batch, output_stats, create_loss, best_score=best_score)


def train_classif_subparts(net, train_set, testset_tuple, labels, criterion, optimizer, best_score=0):
    trans = P.classif_train_trans
    if P.classif_train_pre_proc:
        trans = transforms.Compose([])

    # images are already pre-processed in all cases
    def create_epoch(epoch, train_set, testset_tuple):
        random.shuffle(train_set)
        # labels are needed for stats
        return train_set, {}, {'labels': labels}

    def create_batch(batch, n):
        # must proceed image by image (since different input sizes)
        # each image/batch is composed of multiple scales
        C, H, W = batch[0][0][0].size()
        n_sc = len(batch[0][0])
        train_in = tensor(P.cuda_device, n_sc, C, H, W)
        labels_in = tensor_t(torch.LongTensor, P.cuda_device, n_sc)
        label = labels.index(batch[0][1])
        for j in range(n_sc):
            train_in[j] = trans(batch[0][0][j])
            labels_in[j] = label
        return [train_in], [labels_in]

    def create_loss(t_out, labels_in):
        # t_out contains all subpart classifications for all scales
        loss = criterion(t_out[:, :, 0, 0], labels_in[0])
        for i in range(t_out.size(2)):
            for j in range(t_out.size(3)):
                if i == 0 and j == 0:
                    continue
                loss += criterion(t_out[:, :, i, j], labels_in[0])
        if P.classif_loss_avg:
            loss /= t_out.size(2) * t_out.size(3)
        return loss, None

    train_gen(True, net, train_set, testset_tuple, optimizer, P, create_epoch, create_batch, output_stats, create_loss, best_score=best_score)
