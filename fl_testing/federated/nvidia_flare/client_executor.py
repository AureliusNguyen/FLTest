# fl_testing/federated/nvidia_flare/client_executor.py

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.dxo import DXO, DataKind

import torch
from torch import nn
from torch.optim import SGD
from fl_testing.models.pytorch.simple_network import SimpleNetwork
from fl_testing.data_preprocessing.cifar10_loader import get_cifar10_train_loader

class ClientTrainer(Executor):
    def __init__(self, batch_size=4, epochs=5, lr=0.01):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = SimpleNetwork()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.summary_writer = None

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext):
        # Retrieve the client name
        client_name = fl_ctx.get_identity_name()
        
        # Get the data loader
        train_loader = get_cifar10_train_loader(
            client_name=client_name, batch_size=self.batch_size, shuffle=True
        )
        
        # Load the global model parameters from the DXO
        if shareable:
            # Extract DXO from the shareable
            incoming_dxo = DXO.from_shareable(shareable)
            if incoming_dxo.data_kind == DataKind.WEIGHTS:
                global_weights = incoming_dxo.data
                self.model.load_state_dict(global_weights)
            else:
                self.log_error(fl_ctx, f"Expected DataKind.WEIGHTS but got {incoming_dxo.data_kind}")
                return Shareable().set_return_code(ReturnCode.BAD_TASK_DATA)

        self.model.to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Client: {client_name}, Epoch: {epoch+1}, Loss: {avg_loss}")

        # Collect the updated model parameters
        updated_params = self.model.cpu().state_dict()

        # Create a DXO with the updated weights
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=updated_params)

        # Convert DXO to Shareable
        new_shareable = dxo.to_shareable()

        # Set return code to indicate success
        new_shareable.set_return_code(ReturnCode.OK)

        return new_shareable
