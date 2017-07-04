# -*- encoding: utf-8 -*-

import traceback
import sys
import getopt
import torchvision.transforms as transforms
from model.nn_utils import set_net_train
from utils import *
from train.siamese_regions import P, labels
from train.siamese_regions import get_embeddings, get_siamese_net
from instance_avg import instance_avg


def usage():
    print('Usage: ' + sys.argv[0] + ' [options]')
    prefix = 'Options:\n\tRequired:\n'
    o1 = ('--dataset=\t<path>\tThe path to the dataset containing all ' +
          'reference images. It should contain a sub-folder "test" ' +
          'containing all test images\n')
    o2 = ('--model=\t<name>\tEither AlexNet or ResNet152 to specify the ' +
          'type of model.\n')
    o3 = ('--weights=\t<file>\tThe filename containing weights of a ' +
          'network trained for sub-region classification.\n')
    o4 = ('--device=\t<int>\tThe GPU device used for testing. ' +
          'If negative, CPU is used.\n')
    o5 = ('--feature-dim=\t<int>\tThe feature dimensionality of the network.\n')
    o6 = ('--regions-k=\t<int\tThe number of regions to form the descriptor.\n')
    o7 = ('--dba=\t<int>\tUse DBA with given k. If k = 0, do not use DBA. ' +
          'If k<0, use all neighbors within the same instance.\n')
    o8 = '--help\t\tShow this help\n'
    print(prefix + o1 + o2 + o3 + o4 + o5 + o6 + o7 + o8)


def main(dataset_full, model, weights, device, feature_dim, regions_k, dba):
    # training and test sets
    dataset_id = parse_dataset_id(dataset_full)
    match_labels = match_label_functions[dataset_id]
    train_set_full = get_images_labels(dataset_full, match_labels)
    test_set_full = get_images_labels(dataset_full + '/test', match_labels)

    labels_list = [t[1] for t in train_set_full]
    # setup global params so that testing functions work properly
    labels.extend(sorted(list(set(labels_list))))
    P.num_classes = len(labels)
    P.test_pre_proc = True  # we always pre process images
    P.cuda_device = device
    P.preload_net = weights
    P.cnn_model = model
    P.feature_size2d = feature_sizes[model, image_sizes[dataset_id]]
    P.classif_model = ''  # only useful for training
    P.feature_dim = feature_dim
    P.regions_k = regions_k

    print('Loading and transforming train/test sets.')

    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    m, s = read_mean_std(mean_std_files[dataset_id])
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, s)])
    test_set, test_train_set = [], []
    for im, lab in train_set_full:
        im_o = imread_rgb(im)
        test_train_set.append((test_trans(im_o), lab, im))

    for im, lab in test_set_full:
        if lab not in labels:
            continue
        im_o = imread_rgb(im)
        test_set.append((test_trans(im_o), lab, im))

    print('Testing network on dataset with ID {0}'.format(dataset_id))
    net = get_siamese_net()
    set_net_train(net, False)
    test_embeddings = get_embeddings(net, test_set, device, net.feature_size)
    ref_embeddings = get_embeddings(net, test_train_set, device, net.feature_size)
    sim = torch.mm(test_embeddings, ref_embeddings.t())
    prec1, c, t, _, _ = precision1(sim, test_set, test_train_set)
    mAP = mean_avg_precision(sim, test_set, test_train_set)
    print('Descriptor (TEST): {0} / {1} - acc: {2:.4f} - mAP:{3:.4f}'.format(c, t, prec1, mAP))
    if dba == 0:
        return
    print('Testing using instance feature augmentation')
    dba_embeddings, dba_set = instance_avg(device, ref_embeddings,
                                           test_train_set, labels, dba)
    sim = torch.mm(test_embeddings, dba_embeddings.t())
    prec1, c, t, _, _ = precision1(sim, test_set, dba_set)
    mAP = mean_avg_precision(sim, test_set, dba_set)
    print('Descriptor (TEST DBA k={4}): {0} / {1} - acc: {2:.4f} - mAP:{3:.4f}'.format(c, t, prec1, mAP, dba))


if __name__ == '__main__':
    options_l = (['help', 'dataset=', 'model=', 'weights=', 'device=',
                 'feature-dim=', 'regions-k=', 'dba='])
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', options_l)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    dataset_full, model, weights, device = None, None, None, None
    feature_dim, regions_k, dba = None, None, -1
    for opt, arg in opts:
        if opt in ('--help'):
            usage()
            sys.exit()
        elif opt in ('--dataset'):
            dataset_full = check_folder(arg, 'dataset', True, usage)
        elif opt in ('--model'):
            model = check_model(arg, usage)
        elif opt in ('--weights'):
            weights = check_file(arg, 'initialization weights', True, usage)
        elif opt in ('--device'):
            device = check_int(arg, 'device', usage)
        elif opt in ('--feature-dim'):
            feature_dim = check_int(arg, 'feature-dim', usage)
        elif opt in ('--regions-k'):
            regions_k = check_int(arg, 'regions-k', usage)
        elif opt in ('--dba'):
            dba = check_int(arg, 'dba', usage)
    if (dataset_full is None or model is None or
            weights is None or device is None or
            feature_dim is None or regions_k is None):
        print('One or more required arguments is missing.')
        usage()
        sys.exit(2)

    with torch.cuda.device(device):
        try:
            main(dataset_full, model, weights, device, feature_dim,
                 regions_k, dba)
        except:
            log_detail(P, None, traceback.format_exc())
            raise
