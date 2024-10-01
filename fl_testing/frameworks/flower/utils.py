import torch
from collections import OrderedDict




def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]



