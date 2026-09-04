"""Federated dataset loading & partitioning (built on flwr-datasets).

Supports IID and three non-IID partitioners, which are Dirichlet label skew,
pathological N-classes-per-client, and a natural partition on a real-world client column
such as the FEMNIST writer id. Heterogeneity is what proposal Pitfall-2 (overlooking
dataset sensitivities) and Pitfall-3 (IID-only evaluation) are about.

A dataset that is not named in :data:`DATASET_CONFIG` is treated as a Hugging Face id and
described by inspecting its metadata, so any Hub image-classification dataset is usable
without editing this file.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from diskcache import Index
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    NaturalIdPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from fltest.data.utils import seed_everything


@dataclass(frozen=True)
class DatasetSpec:
    """Everything FLTest needs to know about a dataset to federate it."""

    hf_id: str                      # id passed to flwr-datasets / Hugging Face
    column: str                     # column holding the input (an image)
    label_column: str = "label"     # column holding the class label
    channels: int = 1               # 1 grayscale, 3 RGB
    num_classes: int = 10
    transform: str = "grayscale"    # key into _TRANSFORMS
    natural_partition_by: str = ""  # column giving a real-world client id, if the data has one
    test_split: str = "test"        # split to evaluate on; "" means hold one out of train
    holdout_size: int = 10_000      # examples held out when test_split is ""


DATASET_CONFIG: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", "image", channels=1, num_classes=10, transform="grayscale"),
    "fashion_mnist": DatasetSpec(
        "fashion_mnist", "image", channels=1, num_classes=10, transform="grayscale"
    ),
    "cifar10": DatasetSpec("cifar10", "img", channels=3, num_classes=10, transform="rgb"),
    "cifar100": DatasetSpec(
        "uoft-cs/cifar100", "img", label_column="fine_label",
        channels=3, num_classes=100, transform="rgb",
    ),
    # FEMNIST is the naturally non-IID dataset the proposal calls out: handwritten
    # characters labelled by writer, so `data_distribution: natural` gives each client one
    # real writer instead of a synthetic shard.
    "femnist": DatasetSpec(
        "flwrlabs/femnist", "image", label_column="character",
        channels=1, num_classes=62, transform="grayscale",
        natural_partition_by="writer_id",
        # FEMNIST ships a single train split of 814k examples, so a central test set has to
        # be held out before partitioning. Slicing it out of the client shards instead would
        # evaluate the global model on data its own clients trained on.
        test_split="",
    ),
}

_TRANSFORMS = {
    "rgb": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "grayscale": transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
}

# name -> factory(num_partitions, label_column, **kwargs) -> Partitioner.
# A lower Dirichlet alpha means more label skew across clients.
PARTITIONERS = {
    "iid": lambda n, label_column="label", **kw: IidPartitioner(num_partitions=n),
    "dirichlet": lambda n, label_column="label", alpha=0.5, **kw: DirichletPartitioner(
        num_partitions=n, partition_by=label_column, alpha=alpha
    ),
    "pathological": lambda n, label_column="label", classes_per_partition=2, **kw: (
        PathologicalPartitioner(
            num_partitions=n, partition_by=label_column,
            num_classes_per_partition=classes_per_partition,
        )
    ),
    "natural": lambda n, label_column="label", partition_by="", **kw: NaturalIdPartitioner(
        partition_by=partition_by
    ),
}

#: Fixed seed for the train/test holdout, so a dataset without a test split still gives
#: the same evaluation set on every run and across every framework.
_HOLDOUT_SEED = 786

_probe_cache: Dict[str, DatasetSpec] = {}


def _probe_hub(dataset_name: str) -> DatasetSpec:
    """Describe an unlisted dataset by reading its Hub metadata.

    Only the dataset card and feature schema are fetched, not the data itself, so this is
    cheap. The image column is the first image-valued feature and the label column is the
    first one carrying class names.
    """
    if dataset_name in _probe_cache:
        return _probe_cache[dataset_name]
    from datasets import load_dataset_builder

    features = load_dataset_builder(dataset_name).info.features
    image_col = next(
        (k for k, v in features.items() if type(v).__name__ == "Image"), None
    )
    label_col = next((k for k, v in features.items() if hasattr(v, "names")), None)
    if image_col is None or label_col is None:
        raise ValueError(
            f"Cannot use '{dataset_name}' automatically: FLTest needs one image column and "
            f"one labelled class column, but found {list(features)}. Add an entry to "
            f"DATASET_CONFIG in fltest/data/datasets.py to describe it explicitly."
        )
    num_classes = len(features[label_col].names)
    spec = DatasetSpec(
        dataset_name, image_col, label_column=label_col,
        channels=3, num_classes=num_classes, transform="rgb",
    )
    _probe_cache[dataset_name] = spec
    return spec


def resolve_dataset(dataset_name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for a built-in name or a Hugging Face id."""
    if dataset_name in DATASET_CONFIG:
        return DATASET_CONFIG[dataset_name]
    return _probe_hub(dataset_name)


def list_datasets() -> List[str]:
    return sorted(DATASET_CONFIG)


def list_partitioners() -> List[str]:
    return sorted(PARTITIONERS)


def dataset_meta(dataset_name: str):
    """Return (channels, num_classes) for a built-in dataset or a Hugging Face id."""
    spec = resolve_dataset(dataset_name)
    return spec.channels, spec.num_classes


def get_federated_dataset(dataset_name: str, num_clients: int, partitioner: str = "iid", **part_kwargs):
    """Partition ``dataset_name`` into ``num_clients`` shards (HF datasets, not loaders)."""
    if partitioner not in PARTITIONERS:
        raise ValueError(f"Unknown partitioner '{partitioner}'. Available: {list_partitioners()}")

    spec = resolve_dataset(dataset_name)
    transform = _TRANSFORMS[spec.transform]

    if partitioner == "natural":
        if not spec.natural_partition_by:
            raise ValueError(
                f"'{dataset_name}' has no natural client column, so data_distribution "
                f"'natural' does not apply. Datasets that do: "
                f"{[n for n, s in DATASET_CONFIG.items() if s.natural_partition_by]}."
            )
        part_kwargs = {**part_kwargs, "partition_by": spec.natural_partition_by}

    part = PARTITIONERS[partitioner](num_clients, label_column=spec.label_column, **part_kwargs)

    if spec.test_split:
        fds = FederatedDataset(dataset=spec.hf_id, partitioners={"train": part})
        test_raw = fds.load_split(spec.test_split)
        client_raw = {cid: fds.load_partition(cid) for cid in range(num_clients)}
    else:
        from datasets import load_dataset

        full = load_dataset(spec.hf_id, split="train")
        holdout = full.train_test_split(
            test_size=min(spec.holdout_size, len(full) // 10), seed=_HOLDOUT_SEED, shuffle=True
        )
        part.dataset = holdout["train"]  # partition only the training portion
        test_raw = holdout["test"]
        client_raw = {cid: part.load_partition(cid) for cid in range(num_clients)}

    def apply_transform(img):
        return {"img": transform(img)}

    def prepare(split):
        split = split.map(apply_transform, input_columns=spec.column)
        # Every downstream consumer reads batch["label"], so normalise the label column
        # name here rather than teaching the training loops about each dataset.
        if spec.label_column != "label":
            split = split.rename_column(spec.label_column, "label")
        return split.with_format("torch")

    return {
        "c2data": {cid: prepare(raw) for cid, raw in client_raw.items()},
        "test_data": prepare(test_raw),
    }


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
