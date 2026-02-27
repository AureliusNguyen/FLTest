
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents


from fl_testing.frameworks.models import get_pytorch_model
from fl_testing.frameworks.flower.utils import get_parameters


from fl_testing.frameworks.utils import seed_every_thing


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    # print(">>   ------------------- Clients Metrics ------------- ")
    # all_logs = {}
    # for nk_points, metric_d in metrics:
    #     cid = metric_d["cid"]
    #     temp_s = (
    #         f' Client {metric_d["cid"]}, before_train: {metric_d["before_train"]}, "after_train":{metric_d["after_train"]}'
    #     )
    #     all_logs[cid] = temp_s

    # # sorted by client id from lowest to highest
    # for k in sorted(all_logs.keys()):
    #     print(all_logs[k])

    return {"loss temp": 0.0, "accuracy-temp": 0.0}


def get_server_app(cfg, central_eval_fn):
    seed_every_thing(cfg.seed)
    net2 = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                             deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)

    initial_parameters = get_parameters(net2)

    def _fit_config(server_round: int):
        return {"server_round": server_round}

    def server_fn(context):
        # seed_every_thing(cfg.seed)
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=central_eval_fn,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
            on_fit_config_fn=_fit_config,
            initial_parameters=ndarrays_to_parameters(initial_parameters)

        )
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)

    return server_app
