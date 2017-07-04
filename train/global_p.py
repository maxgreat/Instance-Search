# -*- encoding: utf-8 -*-

# from utils import trans_str, fun_str


def match_label_fou_clean2(x):
    s = x.split('/')[-1].split('_')
    return s[0] + s[1]


def match_label_video(x):
    return x.split('/')[-1].split('-')[0]


def match_label_oxford(x):
    return x.split('/')[-1].split('_')[0]


image_sizes = {
    'CLICIDE': (3, 224, 224),
    'CLICIDE_max_224sq': (3, 224, 224),
    'CLICIDE_video_227sq': (3, 227, 227),
    'CLICIDE_video_224sq': (3, 224, 224),
    'CLICIDE_video_384': (3, 224, 224),
    'CLICIDE_video_448': (3, 224, 224),
    'fourviere_clean2_224sq': (3, 224, 224),
    'fourviere_clean2_384': (3, 224, 224),
    'fourviere_clean2_448': (3, 224, 224),
    'oxford5k_video_224sq': (3, 224, 224),
    'oxford5k_video_384': (3, 224, 224)
}

num_classes = {
    'CLICIDE': 464,
    'CLICIDE_max_224sq': 464,
    'CLICIDE_video_227sq': 464,
    'CLICIDE_video_224sq': 464,
    'CLICIDE_video_384': 464,
    'CLICIDE_video_448': 464,
    'fourviere_clean2_224sq': 311,
    'fourviere_clean2_384': 311,
    'fourviere_clean2_448': 311,
    'oxford5k_video_224sq': 17,
    'oxford5k_video_384': 17
}

feature_sizes = {
    ('alexnet', (3, 224, 224)): (6, 6),
    ('resnet152', (3, 224, 224)): (7, 7),
    ('resnet152', (3, 227, 227)): (8, 8)
}

flat_feature_sizes = {
    ('alexnet', (3, 224, 224)): 9216,
    ('resnet152', (3, 224, 224)): 2048
}

mean_std_files = {
    'CLICIDE': 'data/CLICIDE_224sq_train_ms.txt',
    'CLICIDE_video_227sq': 'data/cli.txt',
    'CLICIDE_video_224sq': 'data/CLICIDE_224sq_train_ms.txt',
    'CLICIDE_max_224sq': 'data/CLICIDE_224sq_train_ms.txt',
    'CLICIDE_video_384': 'data/CLICIDE_384_train_ms.txt',
    'CLICIDE_video_448': 'data/CLICIDE_448_train_ms.txt',
    'fourviere_clean2_224sq': 'data/fourviere_224sq_train_ms.txt',
    'fourviere_clean2_384': 'data/fourviere_384_train_ms.txt',
    'fourviere_clean2_448': 'data/fourviere_448_train_ms.txt',
    'oxford5k_video_224sq': 'data/oxford5k_224sq_train_ms.txt',
    'oxford5k_video_384': 'data/oxford5k_384_train_ms.txt',
}

match_label_functions = {
    'CLICIDE': match_label_video,
    'CLICIDE_video_227sq': match_label_video,
    'CLICIDE_max_224sq': match_label_video,
    'CLICIDE_video_224sq': match_label_video,
    'CLICIDE_video_384': match_label_video,
    'CLICIDE_video_448': match_label_video,
    'fourviere_clean2_224sq': match_label_fou_clean2,
    'fourviere_clean2_384': match_label_fou_clean2,
    'fourviere_clean2_448': match_label_fou_clean2,
    'oxford5k_video_224sq': match_label_oxford,
    'oxford5k_video_384': match_label_oxford
}
