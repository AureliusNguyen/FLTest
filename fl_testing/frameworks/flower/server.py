from diskcache import Index

from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.client import ClientApp

from fl_testing.frameworks.flower.client import FlowerClient
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework
from fl_testing.frameworks.models import get_pytorch_model, sum_model_weights_pytorch, test
from fl_testing.frameworks.flower.utils import set_parameters, get_parameters


from fl_testing.frameworks.utils  import seed_every_thing, test_case_own_gm_model_summation, get_final_round_results
from flwr.common import ndarrays_to_parameters



def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    print( ">>   ------------------- Clients Metrics ------------- ")
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



############ Simulation ####################

def run_flower_simulation(cfg):
    seed_every_thing(cfg.seed)

    final_round_loss = -1
    final_round_accuracy = -1
    sum_of_weights = -1
    own_implmentation_sum_of_weights = -1

    def _central_evaluate(server_round, parameters, config):
        nonlocal final_round_loss, final_round_accuracy, sum_of_weights, own_implmentation_sum_of_weights
        #seed_every_thing(cfg.seed)
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
          own_implmentation_sum_of_weights  = test_case_own_gm_model_summation(cfg)

        return loss, {"accuracy": accuracy}

    fl_dataset_dict = get_dataset_for_framework(cfg)
    test_data_loader = fl_dataset_dict['test_data']
    c2data_loader = fl_dataset_dict['c2data']

    c2batch_sum = fl_dataset_dict['batch_sum']
    net2 = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                                deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)
    
    initial_parameters = get_parameters(net2)

    

    def client_fn(context):  # -> Any:
        #seed_every_thing(cfg.seed)
        partition_id = context.node_config["partition-id"]
        if partition_id < 5:
            client_data = c2data_loader[0]
        else:
            client_data = c2data_loader[1]
        return FlowerClient(client_data, cfg, cid=partition_id).to_client()

    def server_fn(context):
        #seed_every_thing(cfg.seed)
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=_central_evaluate,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
            initial_parameters= ndarrays_to_parameters(initial_parameters)

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
    result = get_final_round_results(final_round_loss, final_round_accuracy, pytorch_gm_sum=own_implmentation_sum_of_weights, framework_gm_sum=sum_of_weights)    
    return result

