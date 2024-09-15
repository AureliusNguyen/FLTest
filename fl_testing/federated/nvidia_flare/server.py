# fl_testing/federated/nvidia_flare/server.py

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

# Import the SimpleNetwork model
from fl_testing.models.pytorch.simple_network import SimpleNetwork

def create_flare_job(n_clients, num_rounds, initial_model):
    """
    Creates a FedAvgJob for Nvidia FLARE.

    Args:
        n_clients (int): Number of clients.
        num_rounds (int): Number of federated learning rounds.
        initial_model (torch.nn.Module): The initial model to distribute to clients.

    Returns:
        FedAvgJob: Configured federated averaging job.
    """
    job = FedAvgJob(
        name="flare_fedavg_job",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
    )

    # Add clients
    train_script = "fl_testing/federated/nvidia_flare/client.py"

    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",  # Add any necessary script arguments here
        )
        job.to(executor, f"site-{i+1}")

    return job

def run_server(job, workspace="temp", gpu="0"):
    """
    Runs the federated learning server.

    Args:
        job (FedAvgJob): The federated averaging job to run.
        workspace (str): Path to the workspace directory.
        gpu (str): GPU device identifier.
    """
    job.simulator_run(workspace=workspace, gpu=gpu)
