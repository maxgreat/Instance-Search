# -*- encoding: utf-8 -*-

from __future__ import print_function
import sys
import tempfile
import inspect
import types
import torch
import os


# to check an option specifying a file that should or should not exist
def check_file(arg, name, should_exist, usage):
    if should_exist and not os.path.isfile(arg):
        print('Cannot find {0} file at path {1}\n'.format(name, arg))
        usage()
        sys.exit(2)
    if not should_exist and os.path.isfile(arg):
        print('Cannot overwrite {0} file at path {1}\n'.format(name, arg))
        usage()
        sys.exit(2)
    return arg


# to check an option specifying a folder that should or should not exist
def check_folder(arg, name, should_exist, usage):
    if should_exist and not os.path.isdir(arg):
        print('Cannot find {0} folder at path {1}\n'.format(name, arg))
        usage()
        sys.exit(2)
    if not should_exist and os.path.isdir(arg):
        print('Cannot overwrite {0} folder at path {1}\n'.format(name, arg))
        usage()
        sys.exit(2)
    return arg


# to check an option specifying the model
def check_model(arg, usage):
    if arg.lower() == 'alexnet' or arg.lower() == 'resnet152':
        return arg.lower()
    print('Model {0} is not a valid model'.format(arg))
    usage()
    sys.exit(2)


# to check an option specifying an integer
def check_int(arg, name, usage):
    try:
        return int(arg)
    except ValueError:
        print('{0} was given as {1}. This is not an integer.\n'
              .format(name, arg))
        usage()
        sys.exit(2)


# to check an option specifying a boolean
def check_bool(arg, name, usage):
    arg = arg.lower()
    if arg == '':
        print('{0} was not given. It should be a boolean (true/yes/y/1 for True and otherwise False).'.format(name))
        usage()
        sys.exit(2)
    if arg == 'true' or arg == 'yes' or arg == 'y' or arg == '1':
        return True
    return False


def parse_dataset_id(dataset_full):
    if dataset_full.endswith('/'):
        dataset_full = dataset_full[:-1]
    return dataset_full.split('/')[-1]


def read_mean_std(fname):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


def fun_str(f):
    if f.__class__ in (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType):
        return f.__name__
    else:
        return f.__class__.__name__


def trans_str(trans):
    return ','.join(fun_str(t) for t in trans.transforms)


def move_device(obj, device):
    if device >= 0:
        return obj.cuda()
    else:
        return obj.cpu()


def tensor_t(t, device, *sizes):
    return move_device(t(*sizes), device)


def tensor(device, *sizes):
    return tensor_t(torch.Tensor, device, *sizes)


def unique_str(P):
    return P.uuid.strftime('%Y%m%d-%H%M%S-%f')


def save(P, f, prefix):
    f.write('{0}\n\n'.format(prefix))
    # for name, value in sorted(vars(P).items()):
    #     if name == 'uuid':
    #         continue
    #     if name in ('test_trans', 'train_trans', 'train_sub_scales'):
    #         if type(value) is list or type(value) is tuple:
    #             value = ', '.join(trans_str(t) for t in value)
    #         else:
    #             value = trans_str(value)
    #     elif name in ('match_labels_f'):
    #         value = fun_str(value)
    #     f.write('{0}:{1}\n'.format(name, value))
    f.write(inspect.getsource(P.__class__))
    f.close()


def save_uuid(P, prefix):
    f = tempfile.NamedTemporaryFile(dir=P.save_dir, delete=False)
    save(P, f, prefix)
    # the following will not work on Windows (would need to add a remove first)
    os.rename(f.name, os.path.join(P.save_dir, unique_str(P) + '.params'))


def log_detail(P, p_file, *args):
    if p_file:
        print(*args, file=p_file)
    if P.log_file:
        with open(P.log_file, 'a') as f:
            print(*args, file=f)


def log(P, *args):
    log_detail(P, sys.stdout, *args)


def mod_param(p_file, param, new_value):
    with open(p_file) as f_in:
        with open(p_file + '.tmp', 'w') as f_out:
            for line in f_in:
                if (line.strip().startswith('self.' + param) and
                        len(line.split('=')) == 2):
                    new_line = (line.split('=')[0] + '= \'' +
                                new_value + '\'\n')
                    f_out.write(new_line)
                else:
                    f_out.write(line)
    # as above, in Windows a remove would be needed first
    os.rename(p_file + '.tmp', p_file)
