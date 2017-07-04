# -*- encoding: utf-8 -*-

from datetime import datetime
from utils import *  # image transforms, general utilities
from global_p import *  # global config

# in AlexNet, there are 5 convolutional layers with parameters
# and 3 FC layers in the classifier
# in ResNet, before first layer, there are 2 modules with parameters.
# then number of blocks per layers:
# ResNet152 - layer 1: 3, layer 2: 8, layer 3: 36, layer 4: 3
# ResNet50 - layer 1: 3, layer 2: 4, layer 3: 6, layer 4: 3
# finally, a single FC layer is used as classifier
untrained_blocks = {
    'alexnet': 4,
    'resnet152': 2 + 3 + 8 + 36
}


# parameters for the sub-regions classification training with AlexNet
class Params(object):

    def __init__(self):
        # general parameters
        self.cnn_model = 'ResNet152'
        self.dataset_full = 'data/pre_proc/CLICIDE_video_448'
        self.cuda_device = 0
        self.dataset_id = parse_dataset_id(self.dataset_full)
        # the file containing mean and standard deviation values
        # for a new dataset, simply use the filename here or add it to the
        # global_p module parameters
        # (this is valid for the following parameters, too)
        self.mean_std_file = mean_std_files[self.dataset_id]
        # the function for obtaining labels from a filename in the dataset
        # this function takes a filename and returns a unique label
        self.match_labels = match_label_functions[self.dataset_id]
        # input size. this is usually always (3, 224, 224) unless larger
        # fixed-size images should be used
        self.image_input_size = image_sizes[self.dataset_id]
        # the number of different labels in the dataset
        self.num_classes = num_classes[self.dataset_id]
        # the 2D size of the convolutional features of the base network
        self.feature_size2d = feature_sizes[(self.cnn_model.lower(), self.image_input_size)]
        # the number of blocks in the base network that should not be trained
        # (starting from the lowest and going to higher layers/blocks)
        # usually, block represents a layer with parameters,
        # for ResNet or equivalent, block is a whole block of layers
        self.untrained_blocks = untrained_blocks[self.cnn_model.lower()]

        # read mean and standard of dataset here to define transforms already
        m, s = read_mean_std(self.mean_std_file)

        # Classification net general and test params
        self.preload_net = ''  # allows to continue training a network
        self.classif_model = 'data/final_classif_sub/cli_resnet152.pth.tar'
        self.test_upfront = True
        self.train = True
        self.test_pre_proc = True
        self.test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, s)])

        # Classification net training params
        self.train_epochs = 20
        self.train_batch_size = 64
        self.train_micro_batch = 1  # has to be 1
        self.train_aug_rot = r = 45
        self.train_aug_hrange = hr = 0
        self.train_aug_vrange = vr = 0
        self.train_aug_hsrange = hsr = 0.25
        self.train_aug_vsrange = vsr = 0.25
        self.train_aug_hflip = hflip = True
        trans = transforms.Compose([random_affine_noisy_cv(rotation=r, h_range=hr, v_range=vr, hs_range=hsr, vs_range=vsr, h_flip=hflip), transforms.ToTensor(), transforms.Normalize(m, s)])

        # transformation for each scale
        self.train_trans = trans
        self.train_pre_proc = False

        self.train_lr = 1e-4
        self.train_momentum = 0.9
        self.train_weight_decay = 0.
        self.train_optim = 'SGD'
        self.train_annealing = {}
        self.train_loss_avg = False
        self.train_loss_int = 10
        self.train_test_int = 0
        # the batch norm layer cannot be trained if the micro-batch size
        # is too small, as global variances/means cannot be properly
        # approximated in this case. so train only when having a batch
        # of at least 16
        self.train_bn = self.train_micro_batch >= 16 or (self.train_micro_batch <= 0 and (self.train_batch_size >= 16 or self.train_batch_size <= 0))

        # Descriptor net parameters
        # if True, test the network as a descriptor
        # (using the normalized classification output):
        self.test_descriptor_net = True
        # the threshold (in Bytes) for embeddings to be computed on GPU
        self.embeddings_cuda_size = 2 ** 30
        self.feature_dim = 2048
        self.regions_k = 6
        self.train_loss2_alpha = 1.0
        self.train_loss2_avg = True
        self.triplet_margin = 0.1
        # number of epochs after which semi-hard triplet choice switches
        # to hard triplet choice
        self.train_epoch_switch = 2

        # UUID for these parameters (current time)
        self.uuid = datetime.now()
        self.save_dir = 'data'
        self.log_file = path.join(self.save_dir, unique_str(self) + '.log')


# global test params:
P = Params()
