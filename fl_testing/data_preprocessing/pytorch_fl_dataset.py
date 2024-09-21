from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from diskcache import Index
from torch.utils.data import DataLoader

# Define transforms for different dataset types
transforms_dict = {
    'rgb': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'grayscale': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

def get_federated_dataset(dataset_name, num_clients, partitioner_config='iid'):
    if partitioner_config == 'iid':
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
        
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
        test_data = test_data.map(lambda img: {"img": transform(img)}, input_columns=img_col_name).with_format("torch")
        
        c2data = {}
        for cid in range(num_clients):
            temp_partition = fds.load_partition(cid)
            torch_partition = temp_partition.map(lambda img: {"img": transform(img)}, input_columns=img_col_name).with_format("torch")
            c2data[cid] = torch_partition
        
        return {'c2data': c2data, 'test_data': test_data}
    else:
        raise ValueError(f"Unknown partitioner config: {partitioner_config}")

def get_cached_federated_dataset(dataset_name, num_clients, cache_path, partitioner_config='iid'):
    dcache = Index(cache_path)
    key = f"{dataset_name}_{num_clients}_{partitioner_config}"
    
    if key not in dcache:
        dcache[key] = get_federated_dataset(dataset_name, num_clients, partitioner_config)
    
    return dcache[key]

def get_dataset_for_framework(cfg):
    if cfg.framework == 'flower':
        dataset_dict = get_cached_federated_dataset(
            cfg.dataset, 
            cfg.DATASET_DIVISION_CLIENTS, 
            cfg.dataset_cache_path, 
            cfg.data_distribution
        )
        
        c2data = {cid: DataLoader(dset, batch_size=cfg.client_batch_size, shuffle=True)
                  for cid, dset in dataset_dict['c2data'].items()}
        test_data = DataLoader(
            dataset_dict['test_data'].select(range(cfg.max_test_data_size)), 
            batch_size=cfg.server_batch_size
        )
        
        return {'test_data': test_data, 'c2data': c2data}
    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")