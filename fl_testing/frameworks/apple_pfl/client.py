import torch
from pfl.model.pytorch import PyTorchModel
from pfl.metrics import Weighted

from fl_testing.frameworks.models import get_pytorch_model, LOSS_FUNCTIONS_PyTorch, OPTIMIZER_PyTorch


def get_pfl_pytorch_model(cfg):
    pytorch_model = get_pytorch_model(cfg.model_name, model_cache_dir=cfg.model_cache_path,
                                      deterministic=cfg.deterministic, channels=cfg.channels, seed=cfg.seed).to(cfg.device)
    loss_fn = LOSS_FUNCTIONS_PyTorch[cfg.loss_fn]()

    def loss(inputs, targets, eval=False):
        return loss_fn(pytorch_model(inputs), targets)

    def metrics(inputs, targets, eval=True):
        with torch.no_grad():
            logits = pytorch_model(inputs)
            loss = loss_fn(logits, targets).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(targets.view_as(pred)).sum().item()
            num_samples = len(inputs)
            return {
                "loss": Weighted(loss, num_samples),
                "accuracy": Weighted(correct, num_samples)
            }

    pytorch_model.loss = loss
    pytorch_model.metrics = metrics

    # print(f"Original model device: {next(pytorch_model.parameters()).device}")

    pfl_pt_model = PyTorchModel(pytorch_model,
                                local_optimizer_create=OPTIMIZER_PyTorch[cfg.optimizer],
                                central_optimizer=OPTIMIZER_PyTorch[cfg.optimizer](pytorch_model.parameters()))

    # print(f"PFL model device: {next(pfl_pt_model.pytorch_model.parameters()).device}")

    # # print(f"PFL pytorch model weights: {pfl_pt_model.pytorch_model.state_dict()}")

    # _ = input("Press Enter to continue...")

    return pfl_pt_model
