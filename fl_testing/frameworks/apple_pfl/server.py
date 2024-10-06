import os
import random
from torch.utils.data import DataLoader

from pfl.aggregate.simulate import SimulatedBackend
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import CentralEvaluationCallback, TrainingProcessCallback
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams
from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.metrics import  Metrics

from fl_testing.frameworks.apple_pfl.client import get_pfl_pytorch_model
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework
from fl_testing.frameworks.models import get_pytorch_model, test, sum_model_weights_pytorch
from fl_testing.frameworks.utils import seed_every_thing, test_case_own_gm_model_summation, get_final_round_results

os.environ['PFL_PYTORCH_DEVICE'] = 'cpu' # Set the device to CPU for PFL . This is a potential bug in the code.


def prepare_pfl_datasets(cfg):
    dataset_dict = get_dataset_for_framework(cfg)
    c2data, central_data = dataset_dict["c2data"], dataset_dict["test_data"]
    
    def user_sampler():
        return random.choice(list(c2data.keys()))

    def make_dataset_fn(user_id):
        inputs, targets = c2data[user_id]["img"], c2data[user_id]["label"]
        return Dataset((inputs, targets), user_id=user_id)

    inputs = central_data['img']   
    targets = central_data['label']
    central_data_pfl = Dataset((inputs, targets))    
    return FederatedDataset(make_dataset_fn, user_sampler), central_data_pfl, dataset_dict


# Custom Callback to Capture Global Weights
class CaptureWeightsCallback(TrainingProcessCallback):
    def __init__(self, model, frequency=1):
        self.model = model
        self.frequency = frequency
        self.global_weights = None

    def after_central_iteration(self, aggregate_metrics, model, *, central_iteration: int):
        if central_iteration % self.frequency == 0:
            print(f"Capturing weights at the end of round {central_iteration}")
            # Save the model state dictionary
            model_weights = model.pytorch_model.state_dict()
            #print(f"Weights extracted: {model_weights}")
            self.global_weights = model_weights
        return False, Metrics()
    
    def get_last_round_gm_weights(self):
        return self.global_weights



def run_pfl_simulation(cfg):
    train_federated_dataset, central_data_pfl, dataset_dict = prepare_pfl_datasets(cfg)
    print("Running PFL simulation")

    # Federated Learning Setup and Run
    simulated_backend = SimulatedBackend(
        training_data=train_federated_dataset,
        val_data=None
    )

    model_train_params = NNTrainHyperParams(
        local_learning_rate=cfg.client_lr,
        local_num_epochs=cfg.client_epochs,
        local_batch_size=cfg.client_batch_size
    )

    print(f"Local batch size: {cfg.client_batch_size}")

    model_eval_params = NNEvalHyperParams(local_batch_size=cfg.client_batch_size)

    algorithm_params = NNAlgorithmParams(
        central_num_iterations=cfg.num_rounds,
        evaluation_frequency=1,
        train_cohort_size=cfg.num_clients,
        val_cohort_size=0
    )

    pfl_nn_model = get_pfl_pytorch_model(cfg)

    # Instantiate the custom callback for capturing weights
    capture_weights_callback = CaptureWeightsCallback(pfl_nn_model, frequency=1)

    callbacks = [
        CentralEvaluationCallback(
            central_data_pfl,
            model_eval_params=model_eval_params,
            frequency=1
        ),
        capture_weights_callback,  # Add the custom callback here
    ]

    fedavg = FederatedAveraging().run(
        algorithm_params=algorithm_params,
        backend=simulated_backend,
        model=pfl_nn_model,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=callbacks
    )
    print("Federated Learning completed!")
    res = construc_result(pfl_nn_model.pytorch_model, dataset_dict['test_data'], cfg)
    
    # gm_weights = capture_weights_callback.get_last_round_gm_weights()
    # final_model = get_pytorch_model(cfg.model_name, model_cache_dir=cfg.model_cache_path,
    #                                   deterministic=cfg.deterministic, channels=cfg.channels, seed=cfg.seed).to(cfg.device)
    # final_model.load_state_dict(gm_weights)
    
    return res


def construc_result(model, test_data, cfg):
    test_loader = DataLoader(test_data.select(range(cfg.max_test_data_size)), batch_size=cfg.server_batch_size, shuffle=False)
    test_loss, test_accuracy = test(net=model, testloader=test_loader, device=cfg.device, loss_fn=cfg.loss_fn)
    sum_of_weights = sum_model_weights_pytorch(model)
    #test_case_pytorch_gm = test_case_own_gm_model_summation(cfg)
    res = get_final_round_results(test_loss, test_accuracy, framework_gm_sum=sum_of_weights)
    return res