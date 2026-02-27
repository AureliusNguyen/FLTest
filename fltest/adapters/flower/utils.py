from collections import OrderedDict
import torch
from fl_testing.frameworks.utils import seed_every_thing


def set_parameters(net, parameters):
    seed_every_thing(786)
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, device=device) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net):
    seed_every_thing(786)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
