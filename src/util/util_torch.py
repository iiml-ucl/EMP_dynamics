"""
Helper functions for PyTorch.

Functions:
    get_trainable_parameters
    save_model_weights

Class:
    Measurement
    PSNR
    SSIM

"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from math import exp
from tqdm import tqdm
import os
import time


def get_gpu_status():
    """
    ref: https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
         https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
    :return:
    """
    # os.system('nvidia-smi -q -d  power >tmp')
    # data = open('tmp', 'r').read()
    # print(data)

    temp_file_name = f'tmp_{datetime.now().microsecond}'
    while os.path.isfile(temp_file_name):
        time.sleep(1)
        temp_file_name = f'tmp_{datetime.now().microsecond}'
    os.system(f'nvidia-smi -q -d  power |grep -A14 GPU|grep Max\ Power\ Limit >{temp_file_name}')
    power_max = [float(x.split()[4]) for x in open(f'{temp_file_name}', 'r').readlines()]
    os.system(f'rm -rf {temp_file_name}')

    temp_file_name = f'tmp_{datetime.now().microsecond}'
    while os.path.isfile(temp_file_name):
        time.sleep(1)
        temp_file_name = f'tmp_{datetime.now().microsecond}'
    os.system(f'nvidia-smi -q -d  power |grep -A14 GPU|grep Avg >{temp_file_name}')
    power_avg = [float(x.split()[2]) for x in open(f'{temp_file_name}', 'r').readlines()]
    os.system(f'rm -rf {temp_file_name}')

    temp_file_name = f'tmp_{datetime.now().microsecond}'
    while os.path.isfile(temp_file_name):
        time.sleep(1)
        temp_file_name = f'tmp_{datetime.now().microsecond}'
    os.system(f'nvidia-smi -q -d memory |grep -A4 GPU|grep Total >{temp_file_name}')
    mem_tot = [float(x.split()[2]) for x in open(f'{temp_file_name}', 'r').readlines()]
    os.system(f'rm -rf {temp_file_name}')

    temp_file_name = f'tmp_{datetime.now().microsecond}'
    while os.path.isfile(temp_file_name):
        time.sleep(1)
        temp_file_name = f'tmp_{datetime.now().microsecond}'
    os.system(f'nvidia-smi -q -d memory |grep -A4 GPU|grep Free >{temp_file_name}')
    mem_free = [float(x.split()[2]) for x in open(f'{temp_file_name}', 'r').readlines()]
    os.system(f'rm -rf {temp_file_name}')

    if mem_free == []:
        temp_file_name = f'tmp_{datetime.now().microsecond}'
        while os.path.isfile(temp_file_name):
            time.sleep(1)
            temp_file_name = f'tmp_{datetime.now().microsecond}'
        os.system(f'nvidia-smi -q -d memory |grep -A4 GPU|grep Used >{temp_file_name}')
        mem_used = [float(x.split()[2]) for x in open(f'{temp_file_name}', 'r').readlines()]
        os.system(f'rm -rf {temp_file_name}')
        mem_free = [i - j for i, j in zip(mem_tot, mem_used)]

    return (mem_tot, mem_free), (power_max, power_avg)


def auto_select_GPU(mode='memory_priority', threshold=0., unit='pct', dwell=5, random_delay=20, candidate_list=None):
    """
    Select GPU automatically.

    :param mode: (optional, String) The method to select GPU.
    :param threshold: (optional, float) Not used yet.
    :param unit: (optional, String) Default: pct (percent)
    :param dwell: (optional, int) Read the GPU status [dwell] times to get the worst status.
    :return: Torch.device object
    """
    # time.sleep(np.random.randint(random_delay))
    mode_options = ['memory_priority',
                    'power_priority',
                    'memory_threshold',
                    'power_threshold']
    assert mode in mode_options, print(f'{datetime.now()} E auto_select_GPU(): Unknown model_options. Select from: '
                                       f'{mode_options}. Get {mode} instead.')

    tq = tqdm(total=dwell, desc=f'GPU Selecting... {mode}:{threshold}:{unit}', unit='dwell', dynamic_ncols=True)

    for i_dwell in range(dwell):
        (mem_tot_new, mem_free_new), (power_max_new, power_avg_new) = get_gpu_status()
        if i_dwell == 0:
            mem_tot, mem_free, power_max, power_avg = mem_tot_new, mem_free_new, power_max_new, power_avg_new
        else:
            mem_free = [min([mem_free[i], mem_free_new[i]]) for i in range(len(mem_tot_new))]
            power_avg = [max([power_avg[i], power_avg_new[i]]) for i in range(len(mem_tot_new))]
        # sleep(1)
        tq.update()
    tq.close()
    power_free = [i-j for (i, j) in zip(power_max, power_avg)]
    if unit.lower() == 'pct':
        pass
    mem_free_pct = [i/j for (i, j) in zip(mem_free, mem_tot)]
    power_free_pct = [i/j for (i, j) in zip(power_free, power_max)]

    # print(mem_free_pct)
    # print(power_free_pct)

    if mode.lower() == 'memory_priority':
        while True:
            i_GPU = np.argmax(mem_free_pct)
            # Solving following issue
            # https://forums.developer.nvidia.com/t/cuda-peer-resources-error-when-running-on-more-than-8-k80s-aws-p2-16xlarge/45351
            if candidate_list is None:
                break
            else:
                if i_GPU in candidate_list:
                    break
                else:
                    mem_free_pct[i_GPU] = 0
            if np.sum(mem_free_pct) == 0:
                exit(f'{datetime.now()} E util_torch.auto_select_GPU(): No GPU available.')

        print(f'{datetime.now()} I Selected GPU: #{i_GPU}. (from 0 to {len(mem_free_pct)-1})')
        device = torch.device(f'cuda:{i_GPU}')
        return device
    else:
        return None


def get_trainable_parameters(model):
    trainable_parameters = {}
    for name, param in model.named_parameters():
        trainable_parameters[name] = param
    return trainable_parameters


def save_model_weights(model, weight_save_path):
    assert weight_save_path.endswith('.npy'), f'{datetime.now()} E The model save path must ends with .npy, get' \
                                              f' {weight_save_path} instead.'
    trainable_parameters = get_trainable_parameters(model)
    np.save(weight_save_path, trainable_parameters)


def load_model_weights(model, weight_load_path, device, para_names=None):
    """

    :param model:
    :param weight_load_path:
    :param device:
    :param para_names:
    :return:
    """
    weight_dict = np.load(weight_load_path, allow_pickle=True).item()
    model_state = model.state_dict()
    if para_names is None:  # Load the whole saved model.
        para_names = weight_dict.keys()
    for para_name in para_names:
        model_state[para_name] = weight_dict[para_name].data.to(device)
    model.load_state_dict(model_state)


def set_trainable(model, para_names, switch):
    """

    :param model:
    :param para_names: (list of String)
    :param switch: (bool) True: trainable, False, not trainable
    :return:
    """
    params = get_trainable_parameters(model)
    for para_name in para_names:
        params[para_name].requires_grad = switch


def select_loss(name='mse'):
    if name.lower() == 'mse' or name.lower() == 'l2':
        # In torch, MSE loss is L2 loss
        return F.mse_loss, nn.MSELoss()
    elif name.lower() == 'cross_entropy':
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        return F.cross_entropy, nn.CrossEntropyLoss
    else:
        exit(f'{datetime.now()} E util_torch.select_loss(): Unknown name. Select from mse, cross_entropy.'
             f' Get {name} instead')


def select_optimizer(name='adam'):
    if name.lower() == 'adam':
        return torch.optim.Adam
    elif name.lower() == 'sgd':
        return torch.optim.SGD
    else:
        exit(f'{datetime.now()} E util_torch.select_optimizer(): Unknown name. Select from adam, sgd.'
             f' Get {name} instead')


def save_ckpt(ckpt_path, epoch, model_obj, optimizer_obj, loss):
    """

    :param epoch:
    :param model_obj:
    :param optimizer_obj:
    :param loss:
    :return:
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_obj.state_dict(),
        'optimizer_state_dict': optimizer_obj.state_dict(),
        'loss': loss,
    }, str(ckpt_path))


def load_ckpt(ckpt_path, model_obj, optimizer_obj, device=None):
    """

    :param device:
    :param ckpt_path:
    :param model_obj:
    :param optimizer_obj:
    :return:
    """
    if device is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
    else:
        ckpt = torch.load(ckpt_path)
    # if device is not None:
    #     model_obj.load_state_dict(ckpt['model_state_dict']).to(device)  # This to.(device) needs to be tested
    # else:
    #     model_obj.load_state_dict(ckpt['model_state_dict'])
    model_obj.load_state_dict(ckpt['model_state_dict'])
    if optimizer_obj is not None:
        # if device is not None:
        #     optimizer_obj.load_state_dict(ckpt['optimizer_state_dict']).to(device)  # This to.(device) needs to be tested
        # else:
        #     optimizer_obj.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_obj.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    loss = ckpt['loss']
    return model_obj, optimizer_obj, epoch, loss


class GridSearch:
    def __init__(self):
        self.para_dict = {}
        self.all_tests = []
        self.length = 0

    def add_para(self, key: str, value_list: list):
        assert key not in self.para_dict.keys()
        self.para_dict[key] = value_list
        self._update()

    def _update(self):
        self.all_tests = list(dict(zip(self.para_dict, x)) for x in itertools.product(*self.para_dict.values()))
        self.length = len(self.all_tests)

    def __len__(self):
        return self.length

    def __call__(self):
        return self.all_tests
