#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset
from mini_model import Linear_model
from gaussian_loss import Gaussian_loss


##  Initialisation.

torch.set_default_dtype(torch.bfloat16)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


##  Load up data.

batch_size_pi = 2

train_data = Pi_dataset("pi_dataset_10000.txt")

train_dataloader = DataLoader(train_data, batch_size = batch_size_pi,
                              shuffle=True)

test_data = Pi_dataset("pi_dataset_2000.txt")

test_dataloader = DataLoader(test_data, batch_size = batch_size_pi,
                             shuffle=True)


##  Define model.

model = Linear_model().to(device)

print(model.state_dict())


##  Define loss.

loss_0 = nn.BCELoss()



##  Define parameter optimisation mechanism.

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(train_dataloader: DataLoader, model: nn.Module, loss_0:
          nn.Module, optimizer: torch.optim.Optimizer):

    ##  Setup parameters for loss function.

    size = len(train_dataloader.dataset)
    
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        
        X, y = X.to(device), y.to(device)

        ##  Compute prediction error.
        
        y_hat = model(X)

        loss_res = loss_0(y_hat, y)
        
        ##  Backpropagation.
        
        loss_res.backward()

        ##  Iterate parameters then reset graph.
        
        optimizer.step()
        
        optimizer.zero_grad()

        if batch % 10000 == 0:

            print(torch.stack([X, y, y_hat], dim=0).T)

            # print(torch.round(y_hat), y)

            print(loss_res.item())
            
            current = (batch + 1) * len(X)
            
            # print(f"[{current:>5d}/{size:>5d}]")

            # print(f"loss_1: {loss_1:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, loss_fn:
         nn.Module):

    ##  Setup parameters for loss function.

    size = len(train_dataloader.dataset)
    
    model.eval()

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    
    correct /= len(dataloader.dataset)
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")    
            
            
epochs = 5

for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    
    train(train_dataloader, model, loss_0, optimizer)
    
    test(test_dataloader, model, loss_0)

print(model.state_dict())

