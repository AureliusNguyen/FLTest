# fl_testing/federated/flower/client.py

from flwr.client import NumPyClient
from fl_testing.frameworks.models import get_pytorch_model, sum_model_weights_pytorch, train, test

from fl_testing.frameworks.flower.utils import set_parameters, get_parameters
from fl_testing.frameworks.utils import seed_every_thing

from diskcache import Index


class FlowerClient(NumPyClient):
    def __init__(self, client_data, cfg, cid):
        seed_every_thing(cfg.seed)
        self.net = get_pytorch_model(cfg.model_name, model_cache_dir=cfg.model_cache_path,
                                     deterministic=cfg.deterministic, channels=cfg.channels, seed=cfg.seed).to(cfg.device)
        self.trainloader = client_data
        self.valloader = client_data
        self.cfg = cfg
        self.cid = cid

    def get_parameters(self, config):
        #seed_every_thing(self.cfg.seed)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        #seed_every_thing(self.cfg.seed)
        #set_parameters(self.net, parameters)

        before_trining_ws = sum_model_weights_pytorch(self.net)

        train(self.net, self.trainloader, epochs=self.cfg.client_epochs, device=self.cfg.device,
              loss_fn=self.cfg.loss_fn, opitmzer_name=self.cfg.optimizer, seed=self.cfg.seed)
        after_trining_ws = sum_model_weights_pytorch(self.net)
        print(
            f'--> cid {self.cid}, before training {before_trining_ws}, after train {after_trining_ws}')

        temp_cache = Index(self.cfg.temp_cache_path)
        temp_cache[f'cid_{self.cid}'] = (
            self.net.state_dict(), len(self.trainloader))

        return get_parameters(self.net), len(self.trainloader), {'cid': self.cid, 'before_train': before_trining_ws, 'after_train': after_trining_ws}

    def evaluate(self, parameters, config):
        #seed_every_thing(self.cfg.seed)
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader,
                              self.cfg.device, loss_fn=self.cfg.loss_fn)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}