"""Federated dataset loading & partitioning (built on flwr-datasets).

Supports IID and two non-IID partitioners (Dirichlet label-skew and pathological
N-classes-per-client) so robustness/metamorphic tests can stress data heterogeneity —
directly addressing proposal Pitfall-2 (overlooking dataset sensitivities) and
Pitfall-3 (IID-only evaluation).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from diskcache import Index
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from fltest.data.utils import seed_everything

# dataset -> (transform key, HF image column name, channels, num_classes)
DATASET_CONFIG = {
    "mnist": ("grayscale", "image", 1, 10),
    "fashion_mnist": ("grayscale", "image", 1, 10),
    "cifar10": ("rgb", "img", 3, 10),
}

_TRANSFORMS = {
    "rgb": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "grayscale": transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
}

# name -> factory(num_partitions) -> Partitioner. alpha low => more non-IID.
PARTITIONERS = {
    "iid": lambda n, **kw: IidPartitioner(num_partitions=n),
    "dirichlet": lambda n, alpha=0.5, **kw: DirichletPartitioner(
        num_partitions=n, partition_by="label", alpha=alpha
    ),
    "pathological": lambda n, classes_per_partition=2, **kw: PathologicalPartitioner(
        num_partitions=n, partition_by="label", num_classes_per_partition=classes_per_partition
    ),
}


def list_datasets() -> List[str]:
    return sorted(DATASET_CONFIG)


def list_partitioners() -> List[str]:
    return sorted(PARTITIONERS)


def dataset_meta(dataset_name: str):
    """Return (channels, num_classes) for a dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list_datasets()}")
    _, _, channels, num_classes = DATASET_CONFIG[dataset_name]
    return channels, num_classes


def get_federated_dataset(dataset_name: str, num_clients: int, partitioner: str = "iid", **part_kwargs):
    """Partition ``dataset_name`` into ``num_clients`` shards (HF datasets, not loaders)."""
    if partitioner not in PARTITIONERS:
        raise ValueError(f"Unknown partitioner '{partitioner}'. Available: {list_partitioners()}")
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list_datasets()}")

    transform_key, img_col, _, _ = DATASET_CONFIG[dataset_name]
    transform = _TRANSFORMS[transform_key]
    part = PARTITIONERS[partitioner](num_clients, **part_kwargs)
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": part})

    def apply_transform(img):
        return {"img": transform(img)}

    test_data = fds.load_split("test").map(apply_transform, input_columns=img_col).with_format("torch")
    c2data = {
        cid: fds.load_partition(cid).map(apply_transform, input_columns=img_col).with_format("torch")
        for cid in range(num_clients)
    }
    return {"c2data": c2data, "test_data": test_data}


def get_cached_federated_dataset(
    dataset_name: str, num_clients: int, cache_path: str, partitioner: str = "iid", **part_kwargs
):
    """Cached wrapper around :func:`get_federated_dataset` keyed by (dataset, n, partitioner, kwargs)."""
    cache = Index(cache_path)
    kw = "_".join(f"{k}{v}" for k, v in sorted(part_kwargs.items()))
    key = f"{dataset_name}_{num_clients}_{partitioner}_{kw}"
    if key not in cache:
        cache[key] = get_federated_dataset(dataset_name, num_clients, partitioner, **part_kwargs)
    return cache[key]


def build_dataloaders(
    dataset_dict: Dict,
    num_clients: int,
    client_batch_size: int,
    server_batch_size: int,
    max_test_size: int,
    seed: int,
):
    """Wrap HF dataset shards in torch DataLoaders (used by reference + Flower backends).

    Only the first ``num_clients`` shards are returned as loaders (the dataset may be
    partitioned more finely than the number of participating clients).
    """
    seed_everything(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    c2loader = {
        cid: DataLoader(
            dataset_dict["c2data"][cid],
            batch_size=client_batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
        )
        for cid in range(num_clients)
    }
    test_split = dataset_dict["test_data"]
    n_test = min(max_test_size, len(test_split))
    test_loader = DataLoader(
        test_split.select(range(n_test)),
        batch_size=server_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return {"c2loader": c2loader, "test_loader": test_loader}


def client_label_counts(dataset_dict: Dict) -> Dict[int, Dict[int, int]]:
    """Per-client class histogram — used by the pitfall checker to detect IID-only setups."""
    counts: Dict[int, Dict[int, int]] = {}
    for cid, data in sorted(dataset_dict["c2data"].items()):
        dset = data.dataset if hasattr(data, "dataset") else data
        labels = dset["label"]
        if torch.is_tensor(labels):
            labels = labels.tolist()
        counts[cid] = dict(Counter(int(x) for x in labels))
    return counts
