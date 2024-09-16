# fl_testing/data_preprocessing/cifar10_loader.py

import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from torchvision import transforms
from flwr_datasets import FederatedDataset

def get_cifar10_train_loader(
    client_name,
    batch_size=32,
    shuffle=True,
    data_root="../data/raw/cifar10/",
    download=True,
):
    """
    Returns a DataLoader for the CIFAR-10 training dataset for a specific client.

    Args:
        client_name (str): Name of the client to customize the data path.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        data_root (str): Root directory where the data is stored.
        download (bool): Whether to download the data if not present.

    Returns:
        DataLoader: DataLoader for the CIFAR-10 training dataset.
    """
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_path = os.path.join(data_root, client_name)

    train_dataset = CIFAR10(
        root=dataset_path, transform=transforms, download=download, train=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader




def flower_cifar10_load_datasets(partition_id, num_clients, batch_size):    
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader
