"""
Activation functions.
"""

import torch.nn as nn


def select_act(name):
    if name.lower() == 'relu':
        return nn.ReLU()
    if name.lower() == 'tanh':
        return nn.Tanh()
    if name.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=1e-2)
    if name.lower() == 'prelu':
        return nn.PReLU()
    if name.lower() == 'elu':
        return nn.ELU()
    if name.lower() == 'softplus':
        return nn.Softplus()
    if name.lower() == 'swish':
        return nn.SiLU()
    if name.lower() == 'sigmoid':
        return nn.Sigmoid()
    if name.lower() == 'softsign':
        return nn.Softsign()




