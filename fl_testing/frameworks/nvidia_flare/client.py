import argparse
import os
import torch

# Import NVFlare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from diskcache import Index

from fl_testing.frameworks.models import get_pytorch_model, train, test
from fl_testing.frameworks.utils import  seed_every_thing
os.environ['PYTHONHASHSEED'] = '786'

seed_every_thing(786)



def main(args):
    seed_every_thing(786)
    client_id = args.client_id

    cache = Index(args.cache_path)

    cfg = cache['flare_cfg']
    dataset_dict = cache['flare_dataset_dict']

    cl_dataset = dataset_dict['c2data'][client_id]

    trainloader = torch.utils.data.DataLoader(
        cl_dataset, batch_size=cfg.client_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset_dict['test_data'].select(range(
        cfg.max_test_data_size)), batch_size=cfg.server_batch_size, num_workers=0, pin_memory=True)

    # Initialize NVFlare client API
    flare.init()

    # Initialize SummaryWriter for TensorBoard metrics
    summary_writer = SummaryWriter()

    # Initialize the network
    net = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                            deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)

    # Training parameters
    epochs = cfg.client_epochs

    while flare.is_running():
        # Receive the global model from the server
        seed_every_thing(786)
        input_model = flare.receive()
        print(
            f"Client {client_id} - Received model for Round {input_model.current_round}")

        # Load the received global model weights
        net.load_state_dict(input_model.params)
        net.train()

        # Perform local training
        net, avg_loss = train(net, trainloader, epochs, cfg.device,
                              cfg.loss_fn, cfg.optimizer, seed=cfg.seed)
        print(f"Client {client_id} - Average Loss: {avg_loss:.3f}")

        # Log training loss
        global_step = input_model.current_round * epochs * len(trainloader)
        summary_writer.add_scalar(
            tag="loss_per_round", scalar=avg_loss, global_step=global_step)

        # Evaluate the model
        loss, accuracy = test(net, testloader, cfg.device, cfg.loss_fn, seed=cfg.seed)
        summary_writer.add_scalar(tag="model_accuracy", scalar=accuracy, global_step=input_model.current_round)

        # Prepare the updated model to send back to the server
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": epochs * len(trainloader)}
        )

        temp_cache = Index(cfg.fw_cache_path)
        temp_cache[f'cid_{args.client_id}'] = (net.state_dict(), len(trainloader))
        
        # Send the updated model back to the server
        flare.send(output_model)
        print(
            f"Client {client_id} - Sent updated model with Accuracy: {accuracy}%")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="NVFlare CIFAR-10 Client")
    parser.add_argument('--client_id', type=int,
                        required=True, help='Unique ID for the client')
    parser.add_argument('--cache_path', type=str,
                        required=True, help='Path to the config cache')
    args2 = parser.parse_args()

    main(args2)
