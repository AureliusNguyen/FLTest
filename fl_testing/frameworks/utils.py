# utils/loss_functions.py
import torch.nn as nn
import torch
import numpy as np
import random

LOSS_FUNCTIONS_PyTorch = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,    
}

OPTIMIZER_PyTorch = {
    'Adam': torch.optim.Adam
}


def sum_model_weights_pytorch(model):
    return sum(p.sum().item() for p in model.parameters())



def train(net, trainloader, epochs, device, loss_fn, opitmzer_name, **args):
    seed_every_thing(seed=args['seed'])    
    
    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    optimizer = OPTIMIZER_PyTorch[opitmzer_name](net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            break 
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

            
        # if verbose:
        #     print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        return net, epoch_loss.item()



def test(net, testloader, device, loss_fn, **args):
    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy






def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




