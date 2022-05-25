"""
[Reusable][PyTorch]

Processors and the wrapper hepler function to pre-process dataset in pytorch way.

Functions and classes should be called in load_dataset.py file.

Created by Zhaoyan @ UCL
"""

import numpy as np
from torch.utils.data import Dataset
import torch
from datetime import datetime

"""
Ref:
    1. Load large npy dataset: https://stackoverflow.com/a/60130903
"""

class Processor_Classification:
    """

    """
    def __init__(self):
        pass

    def __call__(self, input, label, index, train):
        return [input], [label]
'''
PyTorch Input transformers
'''
