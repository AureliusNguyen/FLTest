import torch
import torchvision
from torchvision import transforms
from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset



def _create_federated_mnist(num_users=100, samples_per_user=600):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Distribute data to users
    user_data = {}
    for i in range(num_users):
        start_idx = i * samples_per_user
        end_idx = start_idx + samples_per_user
        user_data[f'user_{i}'] = mnist_train.data[start_idx:end_idx], mnist_train.targets[start_idx:end_idx]
    
    def user_sampler():
        return f"user_{torch.randint(0, num_users, (1,)).item()}"
    
    def make_dataset_fn(user_id):
        inputs, targets = user_data[user_id]
        return Dataset((inputs.float() / 255.0, targets), user_id=user_id)
    
    return FederatedDataset(make_dataset_fn, user_sampler)


def get_fl_mnist(num_clients):
    train_federated_dataset = _create_federated_mnist(num_users=num_clients)
    # Create central test dataset
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    central_data = Dataset((mnist_test.data.float() / 255.0, mnist_test.targets))
    return train_federated_dataset, central_data