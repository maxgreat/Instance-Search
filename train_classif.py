# -*- encoding: utf-8 -*-

import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from test_params import P


# test a classifier model. it should be in eval mode
def test_classif_net(net, test_set, labels, batchSize):
    """
        Test the network accuracy on a test_set
        Return the number of succes and the number of evaluations done
    """
    trans = transforms.Compose([])
    if not P.classif_test_pre_proc:
        trans.transforms.append(P.classif_test_trans)
    if P.test_norm_per_image:
        trans.transforms.append(norm_image_t)

    def eval_batch_test(last, i, is_final, batch):
        correct, total = last
        C, H, W = P.image_input_size
        test_in = tensor(P.cuda_device, len(batch), C, H, W)
        for j, (testIm, _) in enumerate(batch):
            test_in[j] = trans(testIm)

        out = net(Variable(test_in, volatile=True)).data
        _, predicted = torch.max(out, 1)
        total += len(batch)
        correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), test_set, batchSize)


def test_print_classif(net, testset_tuple, labels, best_score=0, epoch=0):
    test_train_set, test_set = testset_tuple
    net.eval()
    c, t = test_classif_net(net, test_set, labels, P.classif_test_batch_size)
    if (c > best_score):
        best_score = c
        prefix = 'CLASSIF, EPOCH:{0}, SCORE:{1}'.format(epoch, c)
        P.save_uuid(prefix)
        torch.save(net, path.join(P.save_dir, P.unique_str() + "_best_classif.ckpt"))
    P.log('TEST - correct:{0}/{1} - acc:{2}'.format(c, t, float(c) / t))

    c, t = test_classif_net(net, test_train_set, labels, P.classif_test_batch_size)
    torch.save(net, path.join(P.save_dir, "model_classif_" + str(epoch) + ".ckpt"))
    P.log("TRAIN - correct:{0}/{1} - acc:{2}".format(c, t, float(c) / t))
    net.train()
    return best_score


def output_stats(net, test_set, epoch, batch_count, is_final, loss, running_loss, score):
    disp_int = P.classif_loss_int
    if batch_count % disp_int == disp_int - 1:
        P.log('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batch_count + 1, running_loss / disp_int))
        running_loss = 0.0

    test_int = P.classif_test_int
    if ((test_int > 0 and batch_count % test_int == test_int - 1) or
            (test_int <= 0 and is_final)):
        score = test_print_classif(net, testset_tuple, labels, score, epoch + 1)
    return running_loss, score


def train_classif(net, train_set, test_set, labels, criterion, optimizer, best_score=0):
    """
        TODO
    """
    C, H, W = P.image_input_size
    trans = P.classif_train_trans
    if P.classif_train_pre_proc:
        trans = transforms.Compose([])

    def create_epoch(epoch, train_set, test_set):
        random.shuffle(train_set)
        return train_set, {}

    def create_batch(batch, n):
        train_in = tensor(P.cuda_device, n, C, H, W)
        lab = tensor_t(torch.LongTensor, P.cuda_device, batchSize)
        for j, (im, lab) in enumerate(batch):
            train_in[j] = trans(im)
            lab[j] = labels.index(lab)
        return [train_in], [lab]

    train_gen(True, net, train_set, test_set, criterion, optimizer, P, create_epoch, create_batch, output_stats, best_score=best_score)
