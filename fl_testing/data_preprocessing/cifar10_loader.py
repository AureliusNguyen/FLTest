# fl_testing/data_preprocessing/cifar10_loader.py

import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

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
