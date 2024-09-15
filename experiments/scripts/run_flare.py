# experiments/scripts/run_flare_experiment.py

import hydra
from omegaconf import DictConfig

# Import the server module
from fl_testing.federated.nvidia_flare.server import (
    create_flare_job,
    run_server,
)
# Import the SimpleNetwork model
from fl_testing.models.pytorch.simple_network import SimpleNetwork

@hydra.main(config_path="../../fl_testing/config", config_name="default")
def main(cfg: DictConfig):
    n_clients = cfg.experiment.num_clients
    num_rounds = cfg.experiment.num_rounds
    initial_model = SimpleNetwork()

    # Create the FLARE job
    job = create_flare_job(
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
    )

    # Run the server
    run_server(job, workspace="temp", gpu="0")

if __name__ == "__main__":
    main()
