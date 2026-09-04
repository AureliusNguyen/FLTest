"""Model zoo + train/test loops shared across FLTest framework adapters.

Models are deterministic when built through :func:`get_model` with ``deterministic=True``:
the initial weights are cached per ``(model, channels, num_classes)``, so every client and
every framework starts from identical parameters. That is a prerequisite for
cross-framework differential testing.

Three kinds of name are accepted. A built-in such as ``LeNet`` comes from
:data:`MODEL_REGISTRY`, a torchvision architecture such as ``ResNet18`` comes from
:data:`TORCHVISION_MODELS`, and ``hf:<id>`` is fetched from the Hugging Face Hub through
timm or transformers (install with ``pip install -e ".[hf]"``).
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

#: FLTest name -> torchvision architecture. torchvision is a core dependency, so these
#: need no extra install. Every one is built from scratch, never with pretrained weights,
#: because federated training starts from a shared random initialisation.
TORCHVISION_MODELS = {
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "ResNet50": "resnet50",
    "VGG11": "vgg11",
    "MobileNetV3": "mobilenet_v3_small",
    "EfficientNetB0": "efficientnet_b0",
}

#: Prefix that routes a model name to the Hugging Face Hub, e.g. ``hf:timm/resnet18``.
HF_PREFIX = "hf:"


def list_models() -> List[str]:
    """Built-in and torchvision names. Hub models are named ``hf:<id>`` and not listed."""
    return sorted(list(MODEL_REGISTRY) + list(TORCHVISION_MODELS))


def _adapt_first_conv(model: nn.Module, channels: int) -> None:
    """Rebuild the first convolution when the dataset has a different channel count.

    torchvision architectures assume 3-channel input, so a grayscale dataset such as MNIST
    or FEMNIST needs the stem replaced. The replacement keeps every other property of the
    original layer.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels == channels:
                return
            replacement = nn.Conv2d(
                channels, module.out_channels, kernel_size=module.kernel_size,
                stride=module.stride, padding=module.padding, bias=module.bias is not None,
            )
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            return


def _build_torchvision(tv_name: str, channels: int, num_classes: int) -> nn.Module:
    from torchvision import models as tv_models

    model = tv_models.get_model(tv_name, weights=None, num_classes=num_classes)
    if tv_name.startswith("resnet"):
        # The ImageNet stem downsamples 7x7 stride 2 then max-pools, which leaves almost
        # nothing of a 32x32 input. Use the standard CIFAR stem instead.
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        _adapt_first_conv(model, channels)
    return model


class _HFImageClassifier(nn.Module):
    """Wrap a transformers image classifier so it returns logits like every other model."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(pixel_values=x).logits


class _HFTextClassifier(nn.Module):
    """Wrap a transformers text classifier so it returns logits like every other model."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits


def _build_hf_text(model_id: str, num_classes: int) -> nn.Module:
    """Build a Hub sequence classifier, randomly initialised for federated training."""
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            f"Loading '{HF_PREFIX}{model_id}' for text needs the Hugging Face extra. Install "
            f'it with pip install -e ".[hf]"'
        ) from exc
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    return _HFTextClassifier(AutoModelForSequenceClassification.from_config(config))


def _build_hf(model_id: str, channels: int, num_classes: int) -> nn.Module:
    """Build a Hub image model by id, trying timm first and then transformers."""
    try:
        import timm
    except ImportError:
        timm = None
    if timm is not None:
        try:
            return timm.create_model(
                model_id.removeprefix("timm/"), pretrained=False,
                num_classes=num_classes, in_chans=channels,
            )
        except Exception:
            pass  # not a timm architecture, so fall through to transformers
    try:
        from transformers import AutoConfig, AutoModelForImageClassification
    except ImportError as exc:
        raise ImportError(
            f"Loading '{HF_PREFIX}{model_id}' needs the Hugging Face extra. Install it with "
            f'pip install -e ".[hf]"'
        ) from exc
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    if hasattr(config, "num_channels"):
        config.num_channels = channels
    return _HFImageClassifier(AutoModelForImageClassification.from_config(config))


def forward_batch(net: nn.Module, batch, device: str):
    """Run one batch through ``net``, for either modality.

    Image batches carry ``img`` and text batches carry ``input_ids`` with
    ``attention_mask``, so this is the single place that knows the difference.
    """
    if "input_ids" in batch:
        return net(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    return net(batch["img"].to(device))


def model_weight_sum(model: nn.Module) -> float:
    """Sum of all parameters — a cheap fingerprint of model state for sanity checks."""
    return sum(p.sum().item() for p in model.parameters())


def _cached_initial_state(
    cache_dir: str, name: str, model: nn.Module, channels: int, num_classes: int
):
    cache = Index(cache_dir)
    # num_classes belongs in the key: the classifier head is sized by it, so caching on
    # (name, channels) alone would hand a 10-class state to a 100-class model.
    key = f"{name}-channels{channels}-classes{num_classes}"
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
    modality: str = "",
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
    # A text dataset reports zero channels, so the modality follows from the data unless a
    # caller states it outright.
    modality = modality or ("text" if channels == 0 else "image")

    if modality == "text":
        if not model_name.startswith(HF_PREFIX):
            raise ValueError(
                f"'{model_name}' is an image model, but this dataset is text. Use a Hugging "
                f"Face sequence classifier such as 'hf:prajjwal1/bert-tiny'."
            )
        model = _build_hf_text(model_name[len(HF_PREFIX):], num_classes)
    elif model_name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_name](channels=channels, num_classes=num_classes)
    elif model_name in TORCHVISION_MODELS:
        model = _build_torchvision(TORCHVISION_MODELS[model_name], channels, num_classes)
    elif model_name.startswith(HF_PREFIX):
        model = _build_hf(model_name[len(HF_PREFIX):], channels, num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list_models()}, or any Hugging Face "
            f"id written as '{HF_PREFIX}<id>'."
        )
    if deterministic:
        if not model_cache_dir:
            raise ValueError("model_cache_dir is required when deterministic=True")
        model.load_state_dict(
            _cached_initial_state(model_cache_dir, model_name, model, channels, num_classes)
        )
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
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(forward_batch(net, batch, device), labels)
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
        labels = batch["label"].to(device)
        outputs = forward_batch(net, batch, device)
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return loss_sum / max(total, 1), correct / max(total, 1)
