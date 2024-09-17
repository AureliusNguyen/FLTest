from pfl.aggregate.simulate import SimulatedBackend
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import CentralEvaluationCallback
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams

from fl_testing.federated.apple_pfl.client import get_pfl_pytorch_model
from fl_testing.data_preprocessing.data_loader import get_fl_mnist



def simulate_fl(cfg):
    train_federated_dataset, central_data = get_fl_mnist(num_clients=cfg.num_clients)
    
    
    # 3. Federated Learning Setup and Run
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
        evaluation_frequency=10,
        train_cohort_size=cfg.clients_per_round,
        val_cohort_size=0
    )

    pfl_nn_model = get_pfl_pytorch_model()



    callbacks = [
        CentralEvaluationCallback(
            central_data,
            model_eval_params=model_eval_params,
            frequency=10
        ),
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