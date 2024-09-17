# fl_testing/federated/flower/server.py

from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.client import ClientApp

from fl_testing.federated.flower.client import FlowerClient
from fl_testing.models.pytorch.lenet import LeNet 
from fl_testing.data_preprocessing.cifar10_loader import flower_cifar10_load_datasets



def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_evaluate_clients=0,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)


def run_flower_simulation(cfg):
    def client_fn(context):
        net = LeNet().to(cfg.device)
        partition_id = context.node_config["partition-id"]
        trainloader, valloader, _ = flower_cifar10_load_datasets(partition_id=partition_id, num_clients=cfg.num_clients, batch_size=cfg.batch_size)
        return FlowerClient(net, trainloader, valloader, cfg.device).to_client()

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if cfg.device == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )
