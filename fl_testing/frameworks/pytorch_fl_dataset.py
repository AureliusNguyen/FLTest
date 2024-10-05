from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from diskcache import Index
from torch.utils.data import DataLoader
from fl_testing.frameworks.utils import seed_every_thing

import numpy as np
import torch

def sum_first_batch(dataloader):
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError(
            "The DataLoader is empty. Cannot compute sum on an empty DataLoader.")

    # Check if the batch is a dictionary
    if isinstance(batch, dict):
        # Extract input tensors assuming they are under the key 'img'
        if "img" not in batch:
            raise KeyError(
                "The batch dictionary does not contain the key 'img'.")
        inputs = batch["img"]
    else:
        # If the batch is a tuple or list, assume the first element is the input tensor
        inputs = batch[0]

    # Compute the sum of all elements in the input tensor
    total_sum = torch.sum(inputs)

    return total_sum.item()


def get_federated_dataset(dataset_name, num_clients, partitioner_config='iid'):
    if partitioner_config == 'iid':
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(dataset=dataset_name, partitioners={
                               "train": partitioner})

        # Determine the appropriate transform based on the dataset
        if dataset_name == "cifar10":
            transform = transforms_dict['rgb']
            img_col_name = 'img'
        elif dataset_name == "mnist":
            transform = transforms_dict['grayscale']
            img_col_name = 'image'
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        test_data = fds.load_split("test")
        test_data = test_data.map(lambda img: {"img": transform(
            img)}, input_columns=img_col_name).with_format("torch")

        c2data = {}
        for cid in range(num_clients):
            temp_partition = fds.load_partition(cid)
            torch_partition = temp_partition.map(lambda img: {"img": transform(
                img)}, input_columns=img_col_name).with_format("torch")
            c2data[cid] = torch_partition

        return {'c2data': c2data, 'test_data': test_data}
    else:
        raise ValueError(f"Unknown partitioner config: {partitioner_config}")


def get_cached_federated_dataset(dataset_name, num_clients, cache_path, partitioner_config='iid'):
    dcache = Index(cache_path)
    key = f"{dataset_name}_{num_clients}_{partitioner_config}"

    if key not in dcache:
        dcache[key] = get_federated_dataset(
            dataset_name, num_clients, partitioner_config)

    return dcache[key]


def get_dataset_for_framework(cfg):
    seed_every_thing(cfg.seed)

    if cfg.framework == 'flower':
        dataset_dict = get_cached_federated_dataset(
            cfg.dataset,
            cfg.DATASET_DIVISION_CLIENTS,
            cfg.dataset_cache_path,
            cfg.data_distribution
        )

        def worker_init_fn(worker_id):
            np.random.seed(cfg.seed + worker_id)

        c2data = {
            cid: DataLoader(
                dset,
                batch_size=cfg.client_batch_size,
                shuffle=True,
                num_workers=0,
                worker_init_fn=worker_init_fn,
                pin_memory=True
            )
            for cid, dset in dataset_dict['c2data'].items()
        }

        c2sum_first_batch = {c: sum_first_batch(
            b) for c, b in c2data.items() if c in list(range(cfg.num_clients))}

        test_data = DataLoader(
            dataset_dict['test_data'].select(range(cfg.max_test_data_size)),
            batch_size=cfg.server_batch_size,
            num_workers=0,
            shuffle=False,  # Ensure no shuffling in test loader
            pin_memory=True
        )
        return {'test_data': test_data, 'c2data': c2data, 'batch_sum': c2sum_first_batch}
    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")
