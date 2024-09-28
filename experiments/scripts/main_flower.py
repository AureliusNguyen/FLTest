from diskcache import Index
import hydra
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.client import ClientApp
import torch.nn as nn
import torch
import numpy as np
import random
from flwr.client import NumPyClient
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
import os
import copy


os.environ['PYTHONHASHSEED'] = '786'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Only if using CUDA


LOSS_FUNCTIONS_PyTorch = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
}

OPTIMIZER_PyTorch = {
    'Adam': torch.optim.Adam
}

def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # Uncommented
    # torch.backends.cudnn.deterministic = True  # Uncommented
    # torch.backends.cudnn.benchmark = False  # Uncommented
    torch.use_deterministic_algorithms(True)  # Add this line


def sum_model_weights_pytorch(model):
    return sum(p.sum().item() for p in model.parameters())

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

def fedavg_aggregate(models_state_dict, num_samples):
    # Ensure the list of models and number of samples have the same length
    assert len(models_state_dict) == len(
        num_samples), "The number of models must match the number of sample counts"

    # Initialize a model with the same architecture as the client models
    global_model_state_dict = copy.deepcopy(models_state_dict[0])

    # Initialize a dictionary to store the weighted sum of parameters
    global_state_dict = {key: torch.zeros_like(
        value) for key, value in global_model_state_dict.items()}

    # Total number of samples across all clients
    total_samples = sum(num_samples)

    # Perform weighted aggregation of the client models
    for state_dict, n in zip(models_state_dict, num_samples):
        # Update global model parameters with the weighted sum
        for key in global_state_dict.keys():
            global_state_dict[key] += state_dict[key] * (n / total_samples)
    return global_state_dict

def _get_weights_from_cache(model_cache_dir, mname, model, channels):
    cache = Index(model_cache_dir)
    cache.clear()
    key = f'{mname}-channels{channels}'
    state_dict = cache.get(key)
    if state_dict is None:
        state_dict = model.state_dict()
        cache[key] = state_dict
    return state_dict


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}



def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    print(">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}
    for nk_points, metric_d in metrics:
        cid = metric_d["cid"]
        temp_s = (
            f' Client {metric_d["cid"]}, before_train: {metric_d["before_train"]}, "after_train":{metric_d["after_train"]}'
        )
        all_logs[cid] = temp_s

    # sorted by client id from lowest to highest
    for k in sorted(all_logs.keys()):
        print(all_logs[k])

    return {"loss": 0.0, "accuracy": 0.0}



class LeNet(nn.Module):
    def __init__(self, channels, num_classes=10):
        """
        Initialize the LeNet model.

        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
        """
        super(LeNet, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)
        # Average pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))               # FC1 -> ReLU
        x = F.relu(self.fc2(x))               # FC2 -> ReLU
        x = self.fc3(x)                       # FC3
        return x




# Define transforms for different dataset types
transforms_dict = {
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


def get_pytorch_model(model_name, model_cache_dir, deterministic, channels, seed):
    seed_every_thing(seed)
    model_name2class = {'LeNet': LeNet}
    if deterministic is None or model_cache_dir is None or seed is None:
        raise ValueError(
            "model_cache_dir must be provided when deterministic is True/False. seed value is also required")

    if model_name not in model_name2class:
        raise ValueError("Model is not defined.")

    model = model_name2class[model_name](channels=channels)  # default
    if deterministic:
        state_dict = _get_weights_from_cache(
            model_cache_dir, model_name, model, channels=channels)
        model.load_state_dict(state_dict)

    return model


def train(net, trainloader, epochs, device, loss_fn, opitmzer_name, **args):
    seed_every_thing(seed=args['seed'])

    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    optimizer = OPTIMIZER_PyTorch[opitmzer_name](net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            break
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        # if verbose:
        #     print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, device, loss_fn, **args):
    seed_every_thing(args['seed'])
    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class FlowerClient(NumPyClient):
    def __init__(self, client_data, cfg, cid):
        seed_every_thing(cfg.seed)
        self.net = get_pytorch_model(cfg.model_name, model_cache_dir=cfg.model_cache_path,
                                     deterministic=cfg.deterministic, channels=cfg.channels, seed=cfg.seed).to(cfg.device)
        self.trainloader = client_data
        self.valloader = client_data
        self.cfg = cfg
        self.cid = cid

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        seed_every_thing(self.cfg.seed)
        set_parameters(self.net, parameters)

        before_trining_ws = sum_model_weights_pytorch(self.net)

        train(self.net, self.trainloader, epochs=self.cfg.client_epochs, device=self.cfg.device,
              loss_fn=self.cfg.loss_fn, opitmzer_name=self.cfg.optimizer, seed=self.cfg.seed)
        after_trining_ws = sum_model_weights_pytorch(self.net)
        print(
            f'--> cid {self.cid}, before training {before_trining_ws}, after train {after_trining_ws}')

        temp_cache = Index(self.cfg.temp_cache_path)
        temp_cache[f'cid_{self.cid}'] = (
            self.net.state_dict(), len(self.trainloader))

        return get_parameters(self.net), len(self.trainloader), {'cid': self.cid, 'before_train': before_trining_ws, 'after_train': after_trining_ws}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader,
                              self.cfg.device, loss_fn=self.cfg.loss_fn)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


############ Simulation ####################

def run_flower_simulation(cfg):

    seed_every_thing(cfg.seed)

    final_round_loss = -1
    final_round_accuracy = -1
    sum_of_weights = -1
    own_implmentation_sum_of_weights = -1

    def _central_evaluate(server_round, parameters, config):
        nonlocal final_round_loss, final_round_accuracy, sum_of_weights, own_implmentation_sum_of_weights
        seed_every_thing(cfg.seed)
        net = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                                deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)
        # Update model with the latest parameters
        set_parameters(net, parameters)
        loss, accuracy = test(
            net, test_data_loader, device=cfg.device, loss_fn=cfg.loss_fn, seed=cfg.seed)
        print(
            f"Server-side evaluation loss {loss} / accuracy {accuracy} in round {server_round}")
        final_round_loss = loss
        final_round_accuracy = accuracy
        sum_of_weights = sum_model_weights_pytorch(net)

        if server_round > 0:
            temp_cache = Index(cfg.temp_cache_path)
            client_weights_nsamples = [
                temp_cache[f'cid_{i}'] for i in range(cfg.num_clients)]
            client_weights = [c[0] for c in client_weights_nsamples]
            client_nsamples = [c[1] for c in client_weights_nsamples]
            gm_state_dict_temp = fedavg_aggregate(
                client_weights, client_nsamples)
            net.load_state_dict(gm_state_dict_temp)

            own_implmentation_sum_of_weights = sum_model_weights_pytorch(net)

        return loss, {"accuracy": accuracy}

    fl_dataset_dict = get_dataset_for_framework(cfg)
    test_data_loader = fl_dataset_dict['test_data']
    c2data_loader = fl_dataset_dict['c2data']

    c2batch_sum = fl_dataset_dict['batch_sum']

    def client_fn(context):  # -> Any:
        seed_every_thing(cfg.seed)
        partition_id = context.node_config["partition-id"]
        if partition_id < 5:
            client_data = c2data_loader[0]
        else:
            client_data = c2data_loader[1]
        return FlowerClient(client_data, cfg, cid=partition_id).to_client()

    def server_fn(context):
        seed_every_thing(cfg.seed)
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=_central_evaluate,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn

        )
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {"client_resources": {
        "num_cpus": 1, "num_gpus": 0.0}, 'init_args': init_args}
    if cfg.device == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )

    return {'c2batch_sum': c2batch_sum, 'final_round_loss': final_round_loss, 'final_round_accuracy': final_round_accuracy, 'own_implmentation_sum_of_weights': own_implmentation_sum_of_weights, 'sum_of_weights': sum_of_weights, }


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    seed_every_thing(cfg.seed)
    key = f"{cfg.exp_name}-{cfg.framework}"

    current_result = run_flower_simulation(cfg)
    cache = Index(cfg.framework_cache_path)
    prev_result = cache.get(key)

    cache[key] = current_result

    if prev_result is not None:
        for k in current_result.keys():
            print(f'{k} -> prev {prev_result[k]}, current {current_result[k]}')

        for k in current_result.keys():
            assert current_result[k] == prev_result[
                k], f"For {k}, Prev : {prev_result[k]}, Current: {current_result[k]}"
            print(f'{k} Passed')


if __name__ == "__main__":
    main()
