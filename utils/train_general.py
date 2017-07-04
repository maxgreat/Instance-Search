# -*- encoding: utf-8 -*-

import gc
import functools
import torch.optim as optim
from torch.autograd import Variable
from model.nn_utils import set_net_train
from general import log


# Generic function to output the stats
def output_stats(train_type, P, test_print, test_net, net, testset_tuple, epoch, batch_count, is_final, loss, running_loss, score):
    disp_int = P.train_loss_int
    running_loss += loss
    if batch_count % disp_int == disp_int - 1:
        log(P, '[{0:d}, {1:5d}] loss: {2:.5f}'.format(epoch + 1, batch_count + 1, running_loss / disp_int))
        running_loss = 0.0
    test_int = P.train_test_int
    if ((test_int > 0 and batch_count % test_int == test_int - 1) or
            (test_int <= 0 and is_final)):
        score = test_print(train_type, P, net, testset_tuple, test_net, score, epoch + 1)
    return running_loss, score


# evaluate a function by batches of size batch_size on the set x
# and fold over the returned values
def fold_batches(f, init, x, batch_size, cut_end=False, add_args={}):
    nx = len(x)
    if batch_size <= 0:
        return f(init, 0, True, x, **add_args)

    def red(last, idx):
        end = min(idx + batch_size, nx)
        if cut_end and idx + batch_size > nx:
            return last
        is_final = end > nx - batch_size if cut_end else end == nx
        return f(last, idx, is_final, x[idx:end], **add_args)
    return functools.reduce(red, range(0, nx, batch_size), init)


def anneal(net, optimizer, epoch, annealing_dict):
    if epoch not in annealing_dict:
        return optimizer
    default_group = optimizer.state_dict()['param_groups'][0]
    lr = default_group['lr'] * annealing_dict[epoch]
    momentum = default_group['momentum']
    weight_decay = default_group['weight_decay']
    return optim.SGD((p for p in net.parameters() if p.requires_grad), lr=lr, momentum=momentum, weight_decay=weight_decay)


def micro_batch_gen(last, i, is_final, batch, P, net, create_batch, batch_args, create_loss):
    gc.collect()
    prev_val, mini_batch_size = last
    n = len(batch)
    tensors_in, labels_in = create_batch(batch, n, **batch_args)
    tensors_out = net(*(Variable(t) for t in tensors_in))
    loss, loss2 = create_loss(tensors_out, [Variable(l) for l in labels_in])
    loss_micro = loss * n / mini_batch_size if P.train_loss_avg else loss
    val = loss_micro.data[0]
    if loss2 is not None:
        loss2_micro = loss2 * n / mini_batch_size if P.train_loss2_avg else loss2
        loss_micro = loss_micro + P.train_loss2_alpha * loss2_micro
        val = val + P.train_loss2_alpha * loss2_micro.data[0]
    loss_micro.backward()
    return prev_val + val, mini_batch_size


def mini_batch_gen(last, i, is_final, batch, train_type, P, test_print, test_net, net, optimizer, testset_tuple, epoch, micro_args):
    batch_count, score, running_loss = last
    optimizer.zero_grad()
    loss, _ = fold_batches(micro_batch_gen, (0.0, len(batch)), batch, P.train_micro_batch, add_args=micro_args)
    optimizer.step()
    running_loss, score = output_stats(train_type, P, test_print, test_net, net, testset_tuple, epoch, batch_count, is_final, loss, running_loss, score)
    return batch_count + 1, score, running_loss


def train_gen(train_type, P, test_print, test_net, net, train_set, testset_tuple, optimizer, create_epoch, create_batch, create_loss, best_score=0):
    set_net_train(net, True, bn_train=P.train_bn)
    for epoch in range(P.train_epochs):
        # annealing
        optimizer = anneal(net, optimizer, epoch, P.train_annealing)

        dataset, batch_args = create_epoch(epoch, train_set, testset_tuple)

        micro_args = {
            'P': P,
            'net': net,
            'create_batch': create_batch,
            'batch_args': batch_args,
            'create_loss': create_loss
        }
        mini_args = {
            'train_type': train_type,
            'P': P,
            'test_print': test_print,
            'test_net': test_net,
            'net': net,
            'optimizer': optimizer,
            'testset_tuple': testset_tuple,
            'epoch': epoch,
            'micro_args': micro_args
        }

        init = 0, best_score, 0.0  # batch count, score, running loss
        _, best_score, _ = fold_batches(mini_batch_gen, init, dataset, P.train_batch_size, cut_end=True, add_args=mini_args)
