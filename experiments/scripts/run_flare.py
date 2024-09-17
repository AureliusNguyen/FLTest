# experiments/scripts/run_flare.py

import hydra
from omegaconf import DictConfig
from fl_testing.federated.nvidia_flare.server import create_flare_job, run_server
from fl_testing.models.pytorch.lenet import LeNet
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

# Import the SimpleNetwork model
from fl_testing.models.pytorch.lenet import LeNet

import sys


sys.path.append('/home/gulzar/Github/fl_frameworks_testing/')

@hydra.main(config_path="../../fl_testing/config", config_name="config")
def main(cfg: DictConfig):
    n_clients = cfg.experiment.num_clients
    num_rounds = cfg.experiment.num_rounds
    initial_model = LeNet()

    train_script = "/home/gulzar/Github/fl_frameworks_testing/fl_testing/federated/nvidia_flare/client.py"



    job = FedAvgJob(
        name="jill_hello_fl", n_clients=n_clients, num_rounds=num_rounds, initial_model=LeNet()
    )

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, f"site-{i+1}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    # job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
    job.simulator_run("temp1", gpu="0")

    

if __name__ == "__main__":
    main()
