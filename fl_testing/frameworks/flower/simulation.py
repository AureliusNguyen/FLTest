from flwr.client import ClientApp
from flwr.simulation import run_simulation

from fl_testing.frameworks.flower.server import get_server_app
from fl_testing.frameworks.flower.client import FlowerClient
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework
from fl_testing.frameworks.utils import seed_every_thing, get_final_round_results


############ Simulation ####################

def run_flower_simulation(cfg):
    seed_every_thing(cfg.seed)

    final_round_loss = -1
    final_round_accuracy = -1
    sum_of_weights = -1
    own_implmentation_sum_of_weights = -1
    fl_dataset_dict = get_dataset_for_framework(cfg)
    c2data_loader = fl_dataset_dict['c2data']

    def client_fn(context):  # -> Any:
        partition_id = context.node_config["partition-id"]
        # if partition_id < 5:
        #     client_data = c2data_loader[0]
        # else:
        #     client_data = c2data_loader[1]
        client_data = c2data_loader[partition_id]
        return FlowerClient(client_data, cfg, cid=partition_id).to_client()

    client_app = ClientApp(client_fn=client_fn)

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {"client_resources": {
        "num_cpus": 1, "num_gpus": 0.0}, 'init_args': init_args}
    if cfg.device == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0

    run_simulation(
        server_app=get_server_app(cfg),
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )
    result = get_final_round_results(final_round_loss, final_round_accuracy,
                                     pytorch_gm_sum=own_implmentation_sum_of_weights, framework_gm_sum=sum_of_weights)
    return result
