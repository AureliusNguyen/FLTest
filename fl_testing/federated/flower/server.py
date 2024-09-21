from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.client import ClientApp

from fl_testing.federated.flower.client import FlowerClient
from fl_testing.data_preprocessing.pytorch_fl_dataset import get_dataset_for_framework
from fl_testing.models.pytorch.models import get_pytorch_model
from fl_testing.federated.flower.utils import set_parameters, test



def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}



def run_flower_simulation(cfg):
    def _central_evaluate(server_round, parameters,config):
        net = get_pytorch_model(cfg.model_name, cfg.model_cache_path, deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, test_data_loader, device=cfg.device, loss_fn=cfg.loss_fn)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy} in round {server_round}")        
        return loss, {"accuracy": accuracy}

    
    
    fl_dataset_dict =  get_dataset_for_framework(cfg)
    test_data_loader = fl_dataset_dict['test_data']
    c2data_loader = fl_dataset_dict['c2data']
  
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        client_data = c2data_loader[partition_id]
        return FlowerClient(client_data, cfg).to_client()

    def server_fn(context):
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn = _central_evaluate,
        )
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}, 'init_args':init_args}
    if cfg.device == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )
