import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from diskcache import Index


from fl_testing.frameworks.models import get_pytorch_model, test, sum_model_weights_pytorch
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework
from fl_testing.frameworks.utils import seed_every_thing, test_case_own_gm_model_summation, get_final_round_results


os.environ['PYTHONHASHSEED'] = '786'

seed_every_thing(786)


class TestFedAvg(FedAvg):
    def run(self):
        seed_every_thing(786)
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds
        # Access temp_cache_path from the FL context

        cache_path = self.get_cache_path()

        cache = Index(cache_path)

        print(f"Cache path: {cache_path}")

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(
                results, aggregate_fn=self.aggregate_fn
            )
            model = self.update_model(model, aggregate_results)
            cache[f'round_{self.current_round}'] = {'gm': model.params}
            # Save the global model weights at the end of each round
            self.save_model(model)

        self.info("Finished FedAvg.")

    def get_cache_path(self):
        # Return the temp_cache_path. You can define it here.
        temp_cache_path = os.environ.get('TEMP_CACHE_PATH')
        if not temp_cache_path:
            raise ValueError(
                "TEMP_CACHE_PATH environment variable is not set.")
        return temp_cache_path

    def get_final_model_state_dict(self):
        cache_path = self.get_cache_path()
        print(f"Getting final model from cache: {cache_path}")
        cache = Index(cache_path)
        final_round = self.num_rounds - 1
        final_model = cache[f'round_{final_round}']['gm']
        return final_model


def run_flare_simulation(cfg):
    dir_path = '/home/gulzar/Github/fl_frameworks_testing/data/flare_working/temp'

    # clear the directory before running the simulation
    os.system(f'rm -rf {dir_path}')
    os.system(f'mkdir -p {dir_path}')

    seed_every_thing(cfg.seed)
    # Define job parameters
    job_name = "cifar10_pt_fedavg"

    # Prepare the dataset and cache it
    dataset_dict = get_dataset_for_framework(cfg)
    cache = Index(cfg.fw_cache_path)
    cache['flare_dataset_dict'] = dataset_dict
    cache['flare_cfg'] = cfg
    num_clients = cfg.num_clients  # Adjust based on the number of clients
    num_rounds = cfg.num_rounds

    os.environ['TEMP_CACHE_PATH'] = cfg.fw_cache_path

    # Define the FedAvg controller with cfg passed
    controller = TestFedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )

    # Initialize the FedJob with the initial global model
    initial_model = get_pytorch_model(
        cfg.model_name,
        cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed
    )
    job = BaseFedJob(
        name=job_name,
        initial_model=initial_model,
    )

    job.to(controller, "server")

    # Initialize each client with a unique client_id
    for client_id in range(num_clients):
        runner = ScriptRunner(
            script=cfg.flare_client_script_path,
            script_args=f"--client_id {client_id} --cache_path {cfg.fw_cache_path}",
        )
        job.to(runner, f"site-{client_id}")

    job.simulator_run(cfg.flare_dir_path)

    # get the final model
    state_dict = controller.get_final_model_state_dict()
    final_model = get_pytorch_model(
        cfg.model_name,
        cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed
    )

    torch_state_dict = {k: torch.from_numpy(v) if isinstance(
        v, np.ndarray) else v for k, v in state_dict.items()}

    final_model.load_state_dict(torch_state_dict)

    # server test loader
    test_loader = DataLoader(dataset_dict['test_data'].select(range(cfg.max_test_data_size)), batch_size=cfg.server_batch_size, shuffle=False)
    test_loss, test_accuracy = test(net=final_model, testloader=test_loader, device=cfg.device, loss_fn=cfg.loss_fn)
    sum_of_weights = sum_model_weights_pytorch(final_model)
    test_case_pytorch_gm = test_case_own_gm_model_summation(cfg)
    res = get_final_round_results(test_loss, test_accuracy, framework_gm_sum=sum_of_weights, pytorch_gm_sum=test_case_pytorch_gm)
    return res
