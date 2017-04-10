# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
import tempfile
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
from os import rename, path
from utils import *


def match_fou_clean2(x):
    s = x.split('/')[-1].split('_')
    return s[0] + s[1]


def match_video(x):
    return x.split('/')[-1].split('-')[0]


def match_oxford(x):
    return x.split('/')[-1].split('_')[0]


image_sizes = {
    'CLICIDE_max_224sq': (3, 224, 224),
    'CLICIDE_video_227sq': (3, 227, 227),
    'CLICIDE_video_224sq': (3, 224, 224),
    'fourviere_clean2_224sq': (3, 224, 224),
    'oxford5k_video_224sq': (3, 224, 224)
}

feature_sizes = {
    (models.alexnet, (3, 224, 224)): (6, 6),
    (models.resnet152, (3, 224, 224)): (7, 7),
    (models.resnet152, (3, 227, 227)): (8, 8)
}

mean_std_files = {
    'CLICIDE_video_227sq': 'data/cli.txt',
    'CLICIDE_video_224sq': 'data/CLICIDE_224sq_train_ms.txt',
    'CLICIDE_max_224sq': 'data/CLICIDE_224sq_train_ms.txt',
    'fourviere_clean2_224sq': 'data/fourviere_224sq_train_ms.txt',
    'oxford5k_video_224sq': 'data/oxford5k_224sq_train_ms.txt'
}

match_image = {
    'CLICIDE_video_227sq': match_video,
    'CLICIDE_max_224sq': match_video,
    'CLICIDE_video_224sq': match_video,
    'fourviere_clean2_224sq': match_fou_clean2,
    'oxford5k_video_224sq': match_oxford
}


# in ResNet, before first layer, there are 2 modules with parameters.
# then number of blocks per layers:
# ResNet152 - layer 1: 3, layer 2: 8, layer 3: 36, layer 4: 3
# ResNet50 - layer 1: 3, layer 2: 4, layer 3: 6, layer 4: 3
# finally, a single FC layer is used as classifier
# in AlexNet, there are 5 convolutional layers with parameters
# and 3 FC layers in the classifier
untrained_blocks = {
    models.alexnet: 4,
    models.resnet152: 2 + 3 + 8 + 36
}


class TestParams(object):

    def __init__(self):

        # UUID for these parameters (current time)
        self.uuid = datetime.now()

        # general parameters
        self.dataset_full = 'data/pre_proc/CLICIDE_video_224sq'
        self.cnn_model = models.resnet152
        self.cuda_device = 1
        self.save_dir = 'data'
        self.dataset_name = self.dataset_full.split('/')[-1].split('_')[0]
        self.dataset_id = self.dataset_full.split('/')[-1]
        self.mean_std_file = mean_std_files[self.dataset_id]
        self.dataset_match_img = match_image[self.dataset_id]
        self.image_input_size = image_sizes[self.dataset_id]
        self.finetuning = True
        self.feature_size2d = feature_sizes[(self.cnn_model, self.image_input_size)]
        self.log_file = path.join(self.save_dir, self.unique_str() + '.log')
        self.test_norm_per_image = False
        # the maximal allowed size in bytes for embeddings on CUDA
        # if the embeddings take more space, move them to CPU
        self.embeddings_cuda_size = 2 ** 30

        self.untrained_blocks = untrained_blocks[self.cnn_model]

        # read mean and standard of dataset here to define transforms already
        m, s = readMeanStd(self.mean_std_file)

        # Classification net general and test params
        self.classif_preload_net = ''
        self.classif_feature_reduc = False
        self.classif_test_upfront = True
        self.classif_train = True
        self.classif_test_batch_size = 32
        self.classif_test_pre_proc = True
        self.classif_test_trans = transforms.Compose([transforms.ToTensor()])
        if not self.test_norm_per_image:
            # normalization not done per image during test
            self.classif_test_trans.transforms.append(transforms.Normalize(m, s))

        # Classification net training params
        self.classif_train_epochs = 50
        self.classif_train_batch_size = 32
        self.classif_train_micro_batch = 0
        self.classif_train_pre_proc = False
        self.classif_train_aug_rot = r = 180
        self.classif_train_aug_hrange = hr = 0
        self.classif_train_aug_vrange = vr = 0
        self.classif_train_aug_hsrange = hsr = 0.75
        self.classif_train_aug_vsrange = vsr = 0.75
        self.classif_train_aug_hflip = hflip = True
        # self.classif_train_trans = transforms.Compose([transforms.Scale(350), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(m, s)])
        self.classif_train_trans = transforms.Compose([random_affine_cv(rotation=r, h_range=hr, v_range=vr, hs_range=hsr, vs_range=vsr, h_flip=hflip), transforms.ToTensor(), transforms.Normalize(m, s)])
        self.classif_lr = 1e-2
        self.classif_momentum = 0.9
        self.classif_weight_decay = 5e-4
        self.classif_optim = 'SGD'
        self.classif_annealing = {30: 0.1}
        self.classif_loss_avg = True
        self.classif_loss_int = 10
        self.classif_test_int = 0

        # settings for feature net constructed from classification net
        self.feature_net_upfront = True
        self.feature_net_use_class_net = True
        self.feature_net_average = False
        self.feature_net_classify = True

        # Siamese net general and testing params
        self.siam_preload_net = ''
        self.siam_test_upfront = False
        self.siam_use_feature_net = True
        self.siam_train = False
        # TODO should this be the number of instances ?
        self.siam_feature_dim = 2048
        self.siam_conv_average = (1, 1)
        self.siam_cos_margin = 0  # 0: pi/2 angle, 0.5: pi/3, sqrt(3)/2: pi/6
        self.siam_test_batch_size = 32
        self.siam_test_pre_proc = True
        self.siam_test_trans = transforms.Compose([transforms.ToTensor()])
        if not self.test_norm_per_image:
            # normalization not done per image during test
            self.siam_test_trans.transforms.append(transforms.Normalize(m, s))

        # Siamese net training params
        # for train mode: 'couples': using cosine loss
        # 'triplets': using triplet loss
        # choice mode: for 'couples':
        # 'rand': using random couples
        # 'hard': using all positives and hardest negative couples
        # for 'triplets':
        # 'rand': using random negatives for all positives
        # 'hard': hard triplets for all positives
        # 'semi-hard': semi-hard triplets for all positives
        # 'easy-hard': easiest positives with hardest negatives
        self.siam_train_mode = 'triplets'
        self.siam_choice_mode = 'semi-hard'

        # general train params
        self.siam_train_trans = self.classif_train_trans
        self.siam_train_pre_proc = False
        self.siam_train_batch_size = 64
        self.siam_train_micro_batch = 8
        self.siam_lr = 1e-3
        self.siam_momentum = 0.9
        self.siam_weight_decay = 0.0
        self.siam_optim = 'SGD'
        self.siam_annealing = {}
        self.siam_train_epochs = 20
        self.siam_loss_avg = False
        self.siam_loss_int = 10
        self.siam_test_int = 0

        # double objective loss params
        self.siam_double_objective = False
        self.siam_do_loss2_alpha = 1.5
        self.siam_do_loss2_avg = True

        # couples params
        self.siam_couples_p = 0.8

        # triplets general params
        self.siam_triplet_margin = 0.2

        # params for semi-hard mode
        # number of epochs after which we
        # take only the hardest examples:
        self.siam_sh_epoch_switch = 2

        # params for easy-hard choice mode
        # n_p: number of easy positives for each image
        # n_n: number of hard negatives for each image
        # n_t: number of hardest triplets actually used for each image
        # Note that if less than n_p positives or n_n negatives exist,
        # we clamp the value to the number of positives/negatives resp.
        # Thus, we must have n_t <= n_p and n_t <= n_n
        self.siam_eh_n_p = 3
        self.siam_eh_n_n = 100
        self.siam_eh_n_t = 3
        self.siam_eh_req_triplets = self.siam_train_batch_size * 64

    def unique_str(self):
        return self.uuid.strftime('%Y%m%d-%H%M%S-%f')

    def save(self, f, prefix):
        f.write('{0}\n'.format(prefix))
        first = ['dataset_full', 'cnn_model', 'classif_train', 'classif_preload_net', 'feature_net_upfront', 'siam_train', 'siam_preload_net']
        for name in first:
            value = getattr(self, name)
            if name == 'cnn_model':
                value = fun_str(value)
            f.write('{0}:{1}\n'.format(name, value))
        f.write('\n')
        for name, value in sorted(vars(self).items()):
            if name == 'uuid' or name in first:
                continue
            if name in ('classif_test_trans', 'classif_train_trans', 'siam_test_trans', 'siam_train_trans'):
                value = trans_str(value)
            elif name in ('cnn_model', 'dataset_match_img'):
                value = fun_str(value)
            f.write('{0}:{1}\n'.format(name, value))
        f.close()

    def save_uuid(self, prefix):
        f = tempfile.NamedTemporaryFile(dir=self.save_dir, delete=False)
        self.save(f, prefix)
        # the following will not work on Windows (would need to add a remove first)
        rename(f.name, path.join(self.save_dir, self.unique_str() + '.params'))

    def log_detail(self, p_file, *args):
        if p_file:
            print(*args, file=p_file)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                print(*args, file=f)

    def log(self, *args):
        self.log_detail(sys.stdout, *args)


# global test params:
P = TestParams()
