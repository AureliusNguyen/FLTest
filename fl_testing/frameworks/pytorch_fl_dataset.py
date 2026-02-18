from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

from fl_testing.frameworks.utils import seed_every_thing

# Dataset configs: transform type and image column name
DATASET_CONFIG = {
    'mnist': ('grayscale', 'image'),
    'cifar10': ('rgb', 'img'),
}

TRANSFORMS = {
    'rgb': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'grayscale': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

# Partitioners: name -> constructor function
# alpha: lower = more non-IID (0.1 extreme, 100 nearly IID)
PARTITIONERS = {
    'iid': lambda n: IidPartitioner(num_partitions=n),
    'dirichlet': lambda n: DirichletPartitioner(num_partitions=n, partition_by="label", alpha=0.5),
    'pathological': lambda n: PathologicalPartitioner(num_partitions=n, partition_by="label", num_classes_per_partition=2),
}


def get_federated_dataset(dataset_name, num_clients, partitioner_config='iid'):
    if partitioner_config not in PARTITIONERS:
        raise ValueError(f"Unknown partitioner: {partitioner_config}. Options: {list(PARTITIONERS.keys())}")
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Options: {list(DATASET_CONFIG.keys())}")

    partitioner = PARTITIONERS[partitioner_config](num_clients)
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

    transform_type, img_col = DATASET_CONFIG[dataset_name]
    transform = TRANSFORMS[transform_type]

    def apply_transform(img):
        return {"img": transform(img)}

    test_data = fds.load_split("test").map(apply_transform, input_columns=img_col).with_format("torch")
    c2data = {
        cid: fds.load_partition(cid).map(apply_transform, input_columns=img_col).with_format("torch")
        for cid in range(num_clients)
    }

    return {'c2data': c2data, 'test_data': test_data}


def get_cached_federated_dataset(dataset_name, num_clients, cache_path, partitioner_config='iid'):
    cache = Index(cache_path)
    key = f"{dataset_name}_{num_clients}_{partitioner_config}"
    if key not in cache:
        cache[key] = get_federated_dataset(dataset_name, num_clients, partitioner_config)
    return cache[key]


def sum_first_batch(dataloader):
    batch = next(iter(dataloader))
    inputs = batch["img"] if isinstance(batch, dict) else batch[0]
    return torch.sum(inputs).item()


def get_dataset_for_framework(cfg):
    seed_every_thing(cfg.seed)

    dataset_dict = get_cached_federated_dataset(
        cfg.dataset, cfg.DATASET_DIVISION_CLIENTS, cfg.dataset_cache_path, cfg.data_distribution
    )

    if cfg.framework in ['flare', 'pfl']:
        return dataset_dict

    # Flower needs DataLoaders
    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)

    c2data = {
        cid: DataLoader(dset, batch_size=cfg.client_batch_size, shuffle=True,
                        num_workers=0, worker_init_fn=worker_init_fn, pin_memory=True)
        for cid, dset in dataset_dict['c2data'].items()
    }

    test_data = DataLoader(
        dataset_dict['test_data'].select(range(cfg.max_test_data_size)),
        batch_size=cfg.server_batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    batch_sums = {c: sum_first_batch(dl) for c, dl in c2data.items() if c < cfg.num_clients}

    return {'test_data': test_data, 'c2data': c2data, 'batch_sum': batch_sums}


def _get_client_label_counts(c2data):
    """Extract per-client label counts from c2data (HF datasets or DataLoaders)."""
    client_counts = {}
    for cid, data in sorted(c2data.items()):
        dset = data.dataset if hasattr(data, 'dataset') else data
        labels = dset['label']
        if torch.is_tensor(labels):
            labels = labels.tolist()
        elif not isinstance(labels, list):
            labels = list(labels)
        client_counts[cid] = dict(Counter(labels))
    return client_counts


def visualize_data_split(dataset_dict, dataset_name, data_distribution, num_clients_show=None, save_path=None):
    """
    Plot how the federated data is split across clients (label distribution).
    Pops up a figure; does not block after closing the figure.
    If save_path is set, saves the figure there (e.g. to tmp/) before showing.
    """
    c2data = dataset_dict['c2data']
    client_counts = _get_client_label_counts(c2data)

    all_labels = sorted(set().union(*(c.keys() for c in client_counts.values())))
    if not all_labels:
        print("No labels found in dataset; skipping visualization.")
        return
    client_ids = sorted(client_counts.keys())
    if num_clients_show is not None:
        client_ids = client_ids[:num_clients_show]
    n_clients = len(client_ids)
    n_labels = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    matrix = np.zeros((n_clients, n_labels), dtype=np.int64)
    for i, cid in enumerate(client_ids):
        for label, count in client_counts[cid].items():
            matrix[i, label_to_idx[label]] = count

    # Sophisticated heatmap styling
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(
        figsize=(max(9, n_labels * 1.0), max(6, n_clients * 0.4)),
        facecolor="#fafafa",
    )
    ax.set_facecolor("#fafafa")

    # Use a refined sequential colormap; show values in cells when readable
    cmap = sns.color_palette("rocket", as_cmap=True)
    annot = n_clients * n_labels <= 200  # annotate when grid is small enough
    fmt = "d"  # integer counts

    im = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Sample count",
            "shrink": 0.8,
            "aspect": 25,
            "pad": 0.02,
        },
        xticklabels=[str(l) for l in all_labels],
        yticklabels=[f"Client {cid}" for cid in client_ids],
        vmin=0,
        vmax=None,
    )

    # Contrast-aware text color for annotations (seaborn may use black/white already)
    if annot and hasattr(im, "collections"):
        # Ensure annotations are readable on dark cells
        for text in ax.texts:
            text.set_fontsize(8)
            text.set_weight("medium")

    ax.set_xlabel("Class label", fontweight="medium")
    ax.set_ylabel("Client", fontweight="medium")
    ax.set_title(
        f"Data split: {dataset_name} | {data_distribution}\n(samples per client × class)",
        fontweight="bold",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved data split visualization to {save_path}")
    plt.show()
