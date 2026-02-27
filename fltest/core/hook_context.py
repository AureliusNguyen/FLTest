"""Mutable context passed to every hook. Only relevant fields are set per hook."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HookContext:
    """Context for hook handlers. Fields are optional; only set what each hook needs."""

    cfg: Optional[Any] = None
    round: Optional[int] = None
    client_id: Optional[int] = None

    # Data phase
    raw_dataset: Optional[Any] = None
    partition_map: Optional[Dict[int, Any]] = None
    dist_dict: Optional[Dict[int, Any]] = None
    test_data: Optional[Any] = None

    # Model / state (Flower uses list of ndarrays for parameters)
    client_data: Optional[Any] = None  # per-client training data/loader
    global_state: Optional[Any] = None
    client_update: Optional[Any] = None
    num_samples: Optional[int] = None
    updates_and_weights: Optional[List[tuple]] = None  # list of (update, n)

    # Outputs
    new_global_state: Optional[Any] = None
    metrics: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = field(default_factory=dict)
