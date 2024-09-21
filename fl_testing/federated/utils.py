# utils/loss_functions.py
import torch.nn as nn
import torch

LOSS_FUNCTIONS_PyTorch = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,    
}

OPTIMIZER_PyTorch = {
    'Adam': torch.optim.Adam
}