from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

from fl_testing.models.pytorch.models import get_pytorch_model
from fl_testing.data_preprocessing.pytorch_fl_dataset import get_dataset_for_framework
from diskcache import Index




def run_simulation(cfg):
    # Define job parameters
    job_name = "cifar10_pt_fedavg"
    initial_model = get_pytorch_model(cfg.model_name, cfg.model_cache_path, deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)
    
    dataset_dict = get_dataset_for_framework(cfg)
    cache = Index(cfg.temp_cache_path)
    cache['flare_dataset_dict'] = dataset_dict
    cache['flare_cfg'] = cfg   
    num_clients = cfg.num_clients  # Adjust based on the number of clients
    num_rounds = cfg.num_rounds

    # Initialize the FedJob with the initial global model
    job = BaseFedJob(
        name=job_name,
        initial_model=initial_model,
    )

    # Define the FedAvg controller
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Initialize each client with a unique client_id
    for client_id in range(0, num_clients):
        runner = ScriptRunner(
            script=cfg.flare_client_script_path,
            script_args=f"--client_id {client_id} --cache_path {cfg.temp_cache_path}",
        )
        job.to(runner, f"site-{client_id}")

    # # Optional: Export the job configuration for external submission
    # job_config_path = cfg.flare_job_path
    # os.makedirs(job_config_path, exist_ok=True)
    # job.export_job(os.path.join(job_config_path, f"{job_name}.job"))

    # print(f"Job '{job_name}' exported to {job_config_path}.")

    # Run the federated learning job in simulation
    # workdir = "/tmp/nvflare/jobs/workdir"
    # os.makedirs(workdir, exist_ok=True)
    job.simulator_run(cfg.flare_dir_path)

    print("Federated learning simulation completed.")

