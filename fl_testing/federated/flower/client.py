# fl_testing/federated/flower/client.py
from re import L
from flwr.client import NumPyClient
from fl_testing.models.pytorch.lenet import LeNet
from fl_testing.federated.flower.utils import set_parameters, get_parameters, train, test

class FlowerClient(NumPyClient):
    def __init__(self, client_data, cfg):
        self.net = LeNet().to(cfg.device)
        self.trainloader = client_data
        self.valloader = client_data
        self.cfg = cfg
        
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.cfg.client_epochs, device=self.cfg.device, loss_fn=self.cfg.loss_fn)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.cfg.device, loss_fn=self.cfg.loss_fn)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
