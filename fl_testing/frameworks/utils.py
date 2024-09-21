# utils/loss_functions.py
import torch.nn as nn
import torch
import numpy as np
import random

LOSS_FUNCTIONS_PyTorch = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,    
}

OPTIMIZER_PyTorch = {
    'Adam': torch.optim.Adam
}


def sum_model_weights_pytorch(model):
    return sum(p.sum().item() for p in model.parameters())







def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




