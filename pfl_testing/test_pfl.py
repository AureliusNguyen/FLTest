import torch
import torchvision
from torchvision import transforms
from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.model.pytorch import PyTorchModel
from pfl.aggregate.simulate import SimulatedBackend
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import CentralEvaluationCallback
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams
from pfl.metrics import Metrics, Weighted

# 1. Data Preparation
def create_federated_mnist(num_users=100, samples_per_user=600):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Distribute data to users
    user_data = {}
    for i in range(num_users):
        start_idx = i * samples_per_user
        end_idx = start_idx + samples_per_user
        user_data[f'user_{i}'] = mnist_train.data[start_idx:end_idx], mnist_train.targets[start_idx:end_idx]
    
    def user_sampler():
        return f"user_{torch.randint(0, num_users, (1,)).item()}"
    
    def make_dataset_fn(user_id):
        inputs, targets = user_data[user_id]
        return Dataset((inputs.float() / 255.0, targets), user_id=user_id)
    
    return FederatedDataset(make_dataset_fn, user_sampler)

train_federated_dataset = create_federated_mnist()

# Create central test dataset
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
central_data = Dataset((mnist_test.data.float() / 255.0, mnist_test.targets))

# 2. Model Definition
class MNISTConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

pytorch_model = MNISTConvNet()
loss_fn = torch.nn.NLLLoss()

def loss(inputs, targets, eval=False):
    return loss_fn(pytorch_model(inputs.unsqueeze(1)), targets)

def metrics(inputs, targets, eval=True):
    with torch.no_grad():
        logits = pytorch_model(inputs.unsqueeze(1))
        loss = loss_fn(logits, targets).item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(targets.view_as(pred)).sum().item()
        num_samples = len(inputs)
        return {
            "loss": Weighted(loss, num_samples),
            "accuracy": Weighted(correct, num_samples)
        }

pytorch_model.loss = loss
pytorch_model.metrics = metrics

model = PyTorchModel(pytorch_model, 
                     local_optimizer_create=torch.optim.SGD,
                     central_optimizer=torch.optim.SGD(pytorch_model.parameters(), 1.0))

# 3. Federated Learning Setup and Run
simulated_backend = SimulatedBackend(
    training_data=train_federated_dataset,
    val_data=None
)

model_train_params = NNTrainHyperParams(
    local_learning_rate=0.01,
    local_num_epochs=5,
    local_batch_size=32
)

model_eval_params = NNEvalHyperParams(local_batch_size=100)

algorithm_params = NNAlgorithmParams(
    central_num_iterations=100,
    evaluation_frequency=10,
    train_cohort_size=10,
    val_cohort_size=0
)

callbacks = [
    CentralEvaluationCallback(
        central_data,
        model_eval_params=model_eval_params,
        frequency=10
    ),
]

model = FederatedAveraging().run(
    algorithm_params=algorithm_params,
    backend=simulated_backend,
    model=model,
    model_train_params=model_train_params,
    model_eval_params=model_eval_params,
    callbacks=callbacks
)

print("Federated Learning completed!")