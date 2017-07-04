# -*- encoding: utf-8 -*-

import torch
import traceback
import time
import datetime
import train.classif_finetune as cf
import train.classif_regions as cr
import train.siamese_regions as sr
from utils import log_detail, mod_param, unique_str
from os import path


# this script simply chains classification fine-tuning (to obtain a
# batch norm model), then classification fine-tuning using sub-regions
# and Siamese fine-tuning using sub-regions
# All training parameters need to be configured in the three config
# files classif_finetune_p.py, classif_regions_p.py and siamese_regions_p.py
# with the exception of the 'bn_model' parameter in classif_regions_p.py
# and the 'classif_model' parameter in siamese_regions_p.py
# These two parameters can simply be left as empty strings as they will
# be over-written by this script
if __name__ == '__main__':
    # make sure UUIDs of train parameter modules are different
    time.sleep(.1)
    cr.P.uuid = datetime.datetime.now()
    cr.P.log_file = path.join(cr.P.save_dir, unique_str(cr.P) + '.log')
    time.sleep(.1)
    sr.P.uuid = datetime.datetime.now()
    sr.P.log_file = path.join(sr.P.save_dir, unique_str(sr.P) + '.log')
    with torch.cuda.device(cf.P.cuda_device):
        try:
            cf.main()
        except:
            log_detail(cf.P, None, traceback.format_exc())
            raise
    p_file = 'train/classif_regions_p.py'
    best_classif = path.join(cf.P.save_dir,
                             unique_str(cf.P) + '_best_classif.pth.tar')
    mod_param(p_file, 'bn_model', best_classif)
    cr.P.bn_model = best_classif
    with torch.cuda.device(cr.P.cuda_device):
        try:
            cr.main()
        except:
            log_detail(cr.P, None, traceback.format_exc())
            raise
    p_file = 'train/siamese_regions_p.py'
    best_classif = path.join(cr.P.save_dir,
                             unique_str(cr.P) + '_best_classif.pth.tar')
    mod_param(p_file, 'classif_model', best_classif)
    sr.P.classif_model = best_classif
    with torch.cuda.device(sr.P.cuda_device):
        try:
            sr.main()
        except:
            log_detail(sr.P, None, traceback.format_exc())
            raise
