"""Label-flipping attack example.

Load via:
  export FLTEST_HOOKS=examples/hooks/atk_label_flip

This marks a single client as "attacker" and flips its local labels before
training each round. After every round it checks whether global accuracy has
dropped compared to round 0 and prints a simple success signal.
"""

from datetime import datetime
from pathlib import Path

import torch

from fltest.core import hooks
from fl_testing.frameworks.models import get_pytorch_model
from fltest.adapters.flower.utils import set_parameters


@hooks.before_client_train
def flip_client_labels(ctx):
    """Flip labels for all clients by rotating them +1 (MNIST-style)."""
    loader = getattr(ctx, "client_data", None)
    if loader is None:
        raise RuntimeError("atk_label_flip: client_data is None for attacker client")

    import torch

    class _LabelFlippedLoader:
        def __init__(self, base):
            self._base = base
            self.dataset = getattr(base, "dataset", None)
            self._num_classes = None
            self._flipped_any = False

        def __len__(self):
            return len(self._base)

        def __iter__(self):
            for batch in self._base:
                if "label" not in batch:
                    raise RuntimeError("atk_label_flip: batch has no 'label' key")
                labels = batch["label"]
                t = torch.as_tensor(labels)
                if self._num_classes is None:
                    self._num_classes = int(t.max().item() + 1)
                    if self._num_classes <= 1:
                        raise RuntimeError(
                            f"atk_label_flip: inferred num_classes={self._num_classes}; "
                            "need at least 2 classes to flip labels"
                        )
                t_flipped = (t + 1) % self._num_classes
                if torch.equal(t, t_flipped):
                    raise RuntimeError("atk_label_flip: labels were not changed by flipping")
                # Preserve original type/device where possible
                if hasattr(labels, "to"):
                    batch["label"] = t_flipped.to(labels.device).type_as(labels)
                else:
                    batch["label"] = t_flipped.tolist()
                self._flipped_any = True
                yield batch

    ctx.client_data = _LabelFlippedLoader(loader)


@hooks.after_round
def validate_label_flip(ctx):
    """After each round, check if predictions follow the +1 label mapping on the test set."""
    if ctx.cfg is None or ctx.global_state is None or ctx.test_data is None:
        return

    cfg = ctx.cfg
    testloader = ctx.test_data

    net = get_pytorch_model(
        cfg.model_name,
        cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed,
    ).to(cfg.device)
    set_parameters(net, ctx.global_state)
    net.eval()

    correct_map, total = 0, 0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(cfg.device), batch["label"].to(cfg.device)
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=1)
            mapped = (labels + 1) % 10  # MNIST-style 10-class rotation
            correct_map += (predicted == mapped).sum().item()
            total += labels.size(0)

    if total == 0:
        return

    mapping_acc = correct_map / total
    success = mapping_acc > 0.2

    msg = f"[ATK] round={ctx.round} mapping_acc={mapping_acc:.4f} (success={success})"
    print(msg)

    base_dir = getattr(cfg, "hook_signal_dir", None)
    log_dir = Path(base_dir if base_dir else "tmp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"atk_label_flip_round_{ctx.round}.txt"
    log_file.write_text(msg + f" at {datetime.now().isoformat()}\n")

