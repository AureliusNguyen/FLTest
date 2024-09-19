from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from diskcache import Index
from torch.utils.data import DataLoader


pytorch_transforms_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def _cifar10_federated(num_clients, partioner_config='iid'):
    if partioner_config == 'iid':
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(dataset="cifar10", partitioners={
                               "train": partitioner})
        test_data = fds.load_split("test")
        test_data = test_data.map(lambda img: {"img": pytorch_transforms_rgb(img)}, input_columns="img").with_format("torch")
        c2data = {}
        for cid in range(num_clients):
            temp_partition = fds.load_partition(cid)
            torch_partition = temp_partition.map(lambda img: {"img": pytorch_transforms_rgb(img)}, input_columns="img").with_format("torch")
            c2data[cid] = torch_partition

        return {'c2data': c2data, 'test_data': test_data}
    else:
        raise ValueError(f"Unknown partitioner config: {partioner_config}")


def _get_federated_dataset(dname, num_clients, cache_path, partioner_config='iid'):
    dcache = Index(cache_path)
    if dname == "cifar10":
        key = f"cifar10_{num_clients}_{partioner_config}"
        if key not in dcache:
            dcache[key] = _cifar10_federated(num_clients, partioner_config)
        return dcache[key]
    else:
        raise ValueError(f"Unknown dataset: {dname}")


def get_dataset_for_framework(cfg):
    DATASET_DIVISION_CLIENTS = cfg.DATASET_DIVISION_CLIENTS
    if cfg.framework == 'flower':
        dataset_dict = _get_federated_dataset(
            cfg.dataset, DATASET_DIVISION_CLIENTS, cfg.dataset_cache_path, cfg.data_distribution)

        c2data = {cid: DataLoader(dset, batch_size=cfg.client_batch_size, shuffle=True)
                  for cid, dset in dataset_dict['c2data'].items()}
        test_data = DataLoader(
            dataset_dict['test_data'], batch_size=cfg.server_batch_size)

        return {'test_data': test_data, 'c2data': c2data}

    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")
