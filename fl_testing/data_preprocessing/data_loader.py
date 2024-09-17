import torch
from torchvision import transforms
from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from flwr_datasets import FederatedDataset as FlwrFederatedDataset


def _create_federated_mnist(num_users=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize Flower's FederatedDataset for MNIST
    fds = FlwrFederatedDataset(
        dataset="cifar10", partitioners={"train": num_users})

    # Function to load data for a specific user
    def make_dataset_fn(user_id):
        user_index = int(user_id.split('_')[1])
        partition = fds.load_partition(user_index)

        # Apply transformations
        def apply_transforms(batch):
            batch["img"] = [transform(img) for img in batch["img"]]
            return batch

        partition = partition.with_transform(apply_transforms)

        # Convert to tensors
        inputs = torch.stack([item['img'] for item in partition]).squeeze()
        targets = torch.tensor([item['label'] for item in partition])

        return Dataset((inputs, targets), user_id=user_id)

    def user_sampler():
        return f"user_{torch.randint(0, num_users, (1,)).item()}"

    return FederatedDataset(make_dataset_fn, user_sampler)


def get_fl_mnist(num_clients):
    train_federated_dataset = _create_federated_mnist(num_users=num_clients)
    # Create central test dataset using Flower's FederatedDataset
    fds = FlwrFederatedDataset(dataset="cifar10", partitioners={
                               "train": num_clients})
    test_data = fds.load_split("test")

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    test_data = test_data.with_transform(apply_transforms)

    # Convert to tensors
    inputs = torch.stack([item['img'] for item in test_data]).squeeze()
    targets = torch.tensor([item['label'] for item in test_data])
    central_data = Dataset((inputs, targets))

    return train_federated_dataset, central_data
