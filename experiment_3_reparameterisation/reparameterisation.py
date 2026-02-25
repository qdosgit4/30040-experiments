#!/usr/bin/env python3

import argparse
import time

import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset
from mini_model_reparam import Linear_model


parser = argparse.ArgumentParser()
parser.add_argument('--neurons', type=int, help='Quantity of n neurons in 1-n-n-1 network.')
args = parser.parse_args()

##  Initialisation.

torch.set_default_dtype(torch.bfloat16)

torch.manual_seed(239852)

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

model = Linear_model(args.neurons).to(device)

##  Save model params to plaintext.

with open("model_params_mu_weight_pre.txt", "w") as f:

    f.write(str(model.state_dict()['linear_relu_stack.0.mu_weight']))
    

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

            # print(loss_res.item())
            
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
    
    print(f"{test_loss:>8f}")
            
            
epochs = 2

# start_time = time.time()

# while time.time() - start_time < 300:
for t in range(epochs):
    
    # print(f"Epoch {t+1}\n-------------------------------")
    
    train(train_dataloader, model, loss_0, optimizer)
    
    test(test_dataloader, model, loss_0)

# print(f"{time.time() - start_time}s")
    
# print(model.state_dict())


##  Save model params to plaintext.

with open("model_params_mu_weight_post.txt", "w") as f:

    f.write(str(model.state_dict()['linear_relu_stack.0.mu_weight']))
    
