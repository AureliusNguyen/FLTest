# fl_testing/federated/flower/client.py
from flwr.client import NumPyClient
from fl_testing.federated.flower.utils import set_parameters, get_parameters, train, test

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1, device=self.device)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
