# fl_testing/federated/nvidia_flare/client.py
import sys
import os
import torch
from torch import nn
from torch.optim import SGD

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
import sys
sys.path.append('/home/gulzar/Github/fl_frameworks_testing/')



# Import the SimpleNetwork model from the new location
from fl_testing.models.pytorch.lenet import LeNet


from fl_testing.federated.utils import LOSS_FUNCTIONS_PyTorch


def main():
    batch_size = 4
    epochs = 5
    lr = 0.01
    model = LeNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = LOSS_FUNCTIONS_PyTorch['CrossEntropyLoss']() # --> Fix this to dynamically get the loss function
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # Get the DataLoader from the data loader module
    train_loader = get_cifar10_train_loader(
        client_name=client_name,
        batch_size=batch_size,
        shuffle=True,
        data_root="data/raw/cifar10/",  # Adjust the path as needed
        download=True,
    )

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss_fn(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
                if i % 3000 == 0 and i > 0:
                    avg_loss = running_loss / 3000
                    print(f"Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {avg_loss}")
                    global_step = (
                        input_model.current_round * steps + epoch * len(train_loader) + i
                    )
                    summary_writer.add_scalar(
                        tag="loss_for_each_batch",
                        scalar_value=avg_loss,
                        global_step=global_step,
                        scalar=running_loss,
                    )
                    running_loss = 0.0

        print("Finished Training")

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
