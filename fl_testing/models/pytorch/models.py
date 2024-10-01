import torch

import torch.nn as nn
import torch.nn.functional as F
from fl_testing.frameworks.utils  import seed_every_thing
from diskcache import Index

"""
First we check if the model is already in the cache, if not we download/randomly initialize it. Save it to the cache and return it.

"""

class LeNet(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        """
        Initialize the LeNet model.

        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)  # Convolutional layer 1
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)      # Average pooling layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)          # Convolutional layer 2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                 # Fully connected layer 1
        self.fc2 = nn.Linear(120, 84)                         # Fully connected layer 2
        self.fc3 = nn.Linear(84, num_classes)                 # Fully connected layer 3

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))               # FC1 -> ReLU
        x = F.relu(self.fc2(x))               # FC2 -> ReLU
        x = self.fc3(x)                       # FC3
        return x



def _get_weights_from_cache(model_cache_dir, mname, model, channels):
    cache = Index(model_cache_dir)
    cache.clear()
    key = f'{mname}-channels{channels}' 
    state_dict = cache.get(key)
    if state_dict is None:
        state_dict = model.state_dict()    
        cache[key] = state_dict 
    return state_dict
   

def get_pytorch_model(model_name, model_cache_dir, deterministic, channels, seed):    
    seed_every_thing(seed)

    model_name2class = {'LeNet': LeNet}

    if deterministic is None or model_cache_dir is None or seed is None:
        raise ValueError("model_cache_dir must be provided when deterministic is True/False. seed value is also required")
    

    if model_name not in model_name2class:
        raise ValueError("Model is not defined.")
   
    torch.manual_seed(seed)
    model = model_name2class[model_name](channels=channels) # default
    if deterministic:
        state_dict =  _get_weights_from_cache(model_cache_dir, model_name, model, channels=channels)
        model.load_state_dict(state_dict)
    
    return model
    

        






