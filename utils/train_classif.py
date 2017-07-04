# -*- encoding: utf-8 -*-

import torch
from model.nn_utils import set_net_train
from os import path
from general import log, save_uuid, unique_str


# Generic function to test and print stats when training a classification net
def test_print_classif(train_type, P, net, testset_tuple, test_net, best_score=0, epoch=0):
    test_set, test_train_set = testset_tuple
    set_net_train(net, False)
    c, t = test_net(net, test_set)
    if (c > best_score):
        best_score = c
        prefix = '{0}, EPOCH:{1}, SCORE:{2}'.format(train_type, epoch, c)
        save_uuid(P, prefix)
        torch.save(net.state_dict(), path.join(P.save_dir, unique_str(P) + "_best_classif.pth.tar"))
    log(P, 'TEST - correct: {0} / {1} - acc: {2}'.format(c, t, float(c) / t))

    c, t = test_net(net, test_train_set)
    torch.save(net.state_dict(), path.join(P.save_dir, "model_classif_" + str(epoch) + ".pth.tar"))
    log(P, 'TRAIN - correct: {0} / {1} - acc: {2}'.format(c, t, float(c) / t))
    set_net_train(net, True, bn_train=P.train_bn)
    return best_score
