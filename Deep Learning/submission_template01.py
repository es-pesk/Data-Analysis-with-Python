import numpy as np
import torch
from torch import nn

def create_model():
    # your code here
    # return model instance (None is just a placeholder)
    NN = nn.Sequential(nn.Linear(784,256 , bias=True),
                   nn.ReLU(),
                   nn.Linear(256, 16, bias=True),
                   nn.ReLU(),
                   nn.Linear(16, 10, bias=True))

    return NN

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    
    # your code here
    params_c = sum(p.numel() for p in model.parameters())

    # верните количество параметров модели model
    return params_c
