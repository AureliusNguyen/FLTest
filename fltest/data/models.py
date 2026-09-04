"""Model zoo + train/test loops shared across FLTest framework adapters.

Models are deterministic when built through :func:`get_model` with ``deterministic=True``:
the initial weights are cached per ``(model, channels)`` so every client/framework starts
from identical parameters — a prerequisite for cross-framework differential testing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diskcache import Index

LOSS_FUNCTIONS = {"CrossEntropyLoss": nn.CrossEntropyLoss}
OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


class LeNet(nn.Module):
    """Classic LeNet-5 for 32x32 inputs (grayscale or RGB)."""

    def __init__(self, channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvNet(nn.Module):
    """Small smooth-activation CNN used for gradient-inversion (DLG) experiments.

    DLG needs twice-differentiable activations, so this uses Sigmoid rather than ReLU
    (matching the canonical "Deep Leakage from Gradients" setup).
    """

    def __init__(self, channels: int = 3, num_classes: int = 10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channels, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MLP(nn.Module):
    """Flatten-then-MLP baseline for quick, cheap smoke tests."""

    def __init__(self, channels: int = 1, num_classes: int = 10, input_hw: int = 32):
        super().__init__()
        in_features = channels * input_hw * input_hw
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "LeNet": LeNet,
    "ConvNet": ConvNet,
    "MLP": MLP,
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def model_weight_sum(model: nn.Module) -> float:
    """Sum of all parameters — a cheap fingerprint of model state for sanity checks."""
    return sum(p.sum().item() for p in model.parameters())


def _cached_initial_state(cache_dir: str, name: str, model: nn.Module, channels: int):
    cache = Index(cache_dir)
    key = f"{name}-channels{channels}"
    state = cache.get(key)
    if state is None:
        state = model.state_dict()
        cache[key] = state
    return state


def get_model(
    model_name: str,
    model_cache_dir: str,
    channels: int,
    num_classes: int = 10,
    deterministic: bool = True,
) -> nn.Module:
    """Instantiate a model, optionally loading cached deterministic initial weights.

    Args:
        model_name: key into :data:`MODEL_REGISTRY`.
        model_cache_dir: diskcache dir for deterministic initial weights.
        channels: input channels (1 grayscale, 3 RGB).
        num_classes: output classes.
        deterministic: if True, load/store a fixed initial state so all
            clients/frameworks start identically.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_models()}")
    model = MODEL_REGISTRY[model_name](channels=channels, num_classes=num_classes)
    if deterministic:
        if not model_cache_dir:
            raise ValueError("model_cache_dir is required when deterministic=True")
        model.load_state_dict(_cached_initial_state(model_cache_dir, model_name, model, channels))
    return model


def train(
    net: nn.Module,
    trainloader,
    epochs: int,
    device: str,
    loss_fn: str = "CrossEntropyLoss",
    optimizer_name: str = "Adam",
    lr: float | None = None,
) -> Tuple[nn.Module, float]:
    """Standard local training loop. Returns (net, last-epoch mean loss)."""
    net.to(device)
    criterion = LOSS_FUNCTIONS[loss_fn]()
    opt_kwargs = {} if lr is None else {"lr": lr}
    optimizer = OPTIMIZERS[optimizer_name](net.parameters(), **opt_kwargs)
    net.train()
    epoch_loss = 0.0
    for _ in range(epochs):
        running, total = 0.0, 0
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            total += labels.size(0)
        epoch_loss = running / max(total, 1)
    return net, epoch_loss


@torch.no_grad()
def test(net: nn.Module, testloader, device: str, loss_fn: str = "CrossEntropyLoss") -> Tuple[float, float]:
    """Evaluate on a loader. Returns (mean loss, accuracy)."""
    net.to(device)
    criterion = LOSS_FUNCTIONS[loss_fn]()
    correct, total, loss_sum = 0, 0, 0.0
    net.eval()
    for batch in testloader:
        images, labels = batch["img"].to(device), batch["label"].to(device)
        outputs = net(images)
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return loss_sum / max(total, 1), correct / max(total, 1)
