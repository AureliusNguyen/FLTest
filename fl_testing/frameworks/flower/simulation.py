from flwr.simulation import run_simulation

from fl_testing.frameworks.flower.server import get_server_app
from fl_testing.frameworks.flower.client import get_client_app
from fl_testing.frameworks.utils import seed_every_thing, get_final_round_results
from fl_testing.frameworks.models import get_pytorch_model, sum_model_weights_pytorch, test
from fl_testing.frameworks.flower.utils import set_parameters
from fl_testing.frameworks.utils import seed_every_thing, test_case_own_gm_model_summation
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework
############ Simulation ####################


def run_flower_simulation(cfg):
    seed_every_thing(cfg.seed)
    final_round_loss = -1
    final_round_accuracy = -1
    sum_of_weights = -1
    own_implmentation_sum_of_weights = -1

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {"client_resources": {
        "num_cpus": 1, "num_gpus": 0.0}, 'init_args': init_args}
    if cfg.device == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0

    def _central_evaluate(server_round, parameters, config):
        nonlocal final_round_loss, final_round_accuracy, sum_of_weights, own_implmentation_sum_of_weights
        # seed_every_thing(cfg.seed)
        net = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                                deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)
        net.to(cfg.device)  # Move model to device (CPU or CUDA)
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
            own_implmentation_sum_of_weights = test_case_own_gm_model_summation(
                cfg)

        return loss, {"accuracy": accuracy}

    fl_dataset_dict = get_dataset_for_framework(cfg)
    test_data_loader = fl_dataset_dict['test_data']

    server_app = get_server_app(cfg, central_eval_fn=_central_evaluate)
    client_app = get_client_app(cfg)

    print('Server app keys', server_app)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )

    result = get_final_round_results(final_round_loss, final_round_accuracy,
                                     pytorch_gm_sum=own_implmentation_sum_of_weights, framework_gm_sum=sum_of_weights)

    return result
