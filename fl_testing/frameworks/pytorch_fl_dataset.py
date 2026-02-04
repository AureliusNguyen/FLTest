from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, PathologicalPartitioner
from diskcache import Index
from torch.utils.data import DataLoader
from fl_testing.frameworks.utils import seed_every_thing

import numpy as np
import torch

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
