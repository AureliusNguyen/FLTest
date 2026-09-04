"""Shared FL utilities: seeding and FedAvg aggregation."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducible runs.

    Determinism is best-effort: it holds on CPU; MPS/CUDA kernels may still be
    non-deterministic, which is why differential tests default to ``device=cpu``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fedavg_aggregate(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    num_samples: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Sample-weighted average of model ``state_dict``s (vanilla FedAvg).

    Args:
        state_dicts: per-client model state dicts (same keys/shapes).
        num_samples: per-client training-set sizes used as weights.

    Returns:
        Aggregated state dict.
    """
    if len(state_dicts) != len(num_samples):
        raise ValueError("state_dicts and num_samples must have equal length")
    if not state_dicts:
        raise ValueError("cannot aggregate an empty list of models")

    total = float(sum(num_samples))
    if total <= 0:
        raise ValueError("total number of samples must be positive")

    agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dicts[0].items()}
    for sd, n in zip(state_dicts, num_samples):
        weight = n / total
        for k in agg:
            agg[k] += sd[k].to(torch.float32) * weight
    # Cast back to the original dtypes.
    return {k: agg[k].to(state_dicts[0][k].dtype) for k in agg}


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """Flatten a ``state_dict`` to an ordered list of numpy arrays.

    This list-of-ndarrays form is the canonical parameter representation shared by every
    backend, so attacks and defenses operate on it identically regardless of framework.
    """
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def load_ndarrays_into(model, arrays: List[np.ndarray]) -> None:
    """Load an ordered list of numpy arrays back into ``model`` (inverse of the above)."""
    import torch as _torch

    keys = list(model.state_dict().keys())
    if len(keys) != len(arrays):
        raise ValueError(f"param count mismatch: model has {len(keys)}, got {len(arrays)}")
    new_state = {k: _torch.as_tensor(a) for k, a in zip(keys, arrays)}
    model.load_state_dict(new_state, strict=True)


def aggregate_ndarrays(
    updates_and_weights: List[tuple],
) -> List[np.ndarray]:
    """Sample-weighted average of parameter lists (list-of-ndarrays form).

    Args:
        updates_and_weights: list of ``(params, num_samples)`` where ``params`` is a
            list of numpy arrays (one per layer).

    Returns:
        Aggregated list of numpy arrays.
    """
    if not updates_and_weights:
        raise ValueError("nothing to aggregate")
    total = float(sum(n for _, n in updates_and_weights))
    num_layers = len(updates_and_weights[0][0])
    out = [np.zeros_like(updates_and_weights[0][0][i], dtype=np.float64) for i in range(num_layers)]
    for params, n in updates_and_weights:
        w = n / total
        for i in range(num_layers):
            out[i] += params[i].astype(np.float64) * w
    return [out[i].astype(updates_and_weights[0][0][i].dtype) for i in range(num_layers)]
