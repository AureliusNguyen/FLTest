import torch
from pfl.model.pytorch import PyTorchModel
from pfl.metrics import  Weighted


from fl_testing.models.pytorch.lenet import LeNet
from fl_testing.federated.utils import LOSS_FUNCTIONS_PyTorch



def get_pfl_pytorch_model(loss_fn):
    pytorch_model = LeNet(channels=3)
    loss_fn = LOSS_FUNCTIONS_PyTorch[loss_fn]()

    def loss(inputs, targets, eval=False):
        return loss_fn(pytorch_model(inputs), targets)

    def metrics(inputs, targets, eval=True):
        with torch.no_grad():
            logits = pytorch_model(inputs)
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



    pt_model = PyTorchModel(pytorch_model, 
                        local_optimizer_create=torch.optim.SGD,
                        central_optimizer=torch.optim.SGD(pytorch_model.parameters(), 1.0))
    return pt_model
