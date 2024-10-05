# utils/loss_functions.py
import torch
import numpy as np
import random
import copy





def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def fedavg_aggregate(models_state_dict, num_samples):
    # Ensure the list of models and number of samples have the same length
    assert len(models_state_dict) == len(
        num_samples), "The number of models must match the number of sample counts"

    # Initialize a model with the same architecture as the client models
    global_model_state_dict = copy.deepcopy(models_state_dict[0])

    # Initialize a dictionary to store the weighted sum of parameters
    global_state_dict = {key: torch.zeros_like(
        value) for key, value in global_model_state_dict.items()}

    # Total number of samples across all clients
    total_samples = sum(num_samples)

    # Perform weighted aggregation of the client models
    for state_dict, n in zip(models_state_dict, num_samples):
        # Update global model parameters with the weighted sum
        for key in global_state_dict.keys():
            global_state_dict[key] += state_dict[key] * (n / total_samples)
    return global_state_dict