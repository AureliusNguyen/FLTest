import types
import torch
from pfl.model.pytorch import PyTorchModel
from pfl.metrics import Weighted

from fl_testing.frameworks.models import get_pytorch_model, LOSS_FUNCTIONS_PyTorch, OPTIMIZER_PyTorch


def get_pfl_pytorch_model(cfg):
    pytorch_model = get_pytorch_model(cfg.model_name, model_cache_dir=cfg.model_cache_path,
                                      deterministic=cfg.deterministic, channels=cfg.channels, seed=cfg.seed).to(cfg.device)
    loss_fn = LOSS_FUNCTIONS_PyTorch[cfg.loss_fn]()

    # Define loss and metrics as proper methods that use `self`
    # This ensures they always use the current model state (updated weights)
    def loss_method(self, inputs, targets, eval=False):
        self.eval() if eval else self.train()
        return loss_fn(self(inputs), targets)

    def metrics_method(self, inputs, targets, eval=True):
        self.eval() if eval else self.train()
        with torch.no_grad():
            logits = self(inputs)
            loss_val = loss_fn(logits, targets).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(targets.view_as(pred)).sum().item()
            num_samples = len(inputs)
            return {
                "loss": Weighted(loss_val, num_samples),
                "accuracy": Weighted(correct, num_samples)
            }

    # Bind methods to the model using types.MethodType so they receive `self`
    pytorch_model.loss = types.MethodType(loss_method, pytorch_model)
    pytorch_model.metrics = types.MethodType(metrics_method, pytorch_model)

    # For FedAvg, central optimizer should use lr=1.0 to directly apply averaged updates
    # (PFL treats model differences as gradients, so lr=1.0 means direct addition)
    pfl_pt_model = PyTorchModel(pytorch_model,
                                local_optimizer_create=OPTIMIZER_PyTorch[cfg.optimizer],
                                central_optimizer=torch.optim.SGD(pytorch_model.parameters(), lr=1.0))

    return pfl_pt_model
