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
from optimiser_sgd_reparam import SGD_reparam


##  Initialisation.

parser = argparse.ArgumentParser()
parser.add_argument('--neurons',
                    required = True,
                    type = int,
                    help = 'Quantity of n neurons in 1-n-n-1 network. Try 512, or 1024.')
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(239852)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def main():

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

    loss = nn.BCELoss()

    ##  Define parameter optimisation mechanism.

    optimizer = SGD_reparam(model.parameters(), lr=1e-3)

    ##  Set quantity of training epochs.

    epochs = 3

    start_time = time.time()
    # while time.time() - start_time < 300:

    ##  Run train-test sequence.
    
    for t in range(epochs):

        print(f"Training...")

        train(train_dataloader, model, loss, optimizer)

        print(f"Testing...")

        test(test_dataloader, model, loss)

    print(f"{time.time() - start_time}s")
    
    # print(model.state_dict())

    ##  Save model params to plaintext.

    with open("model_params_mu_weight_post.txt", "w") as f:

        f.write(str(model.state_dict()['linear_relu_stack.0.mu_weight']))
    

def train(dl: DataLoader, model: nn.Module, loss: nn.Module,
          optimizer: torch.optim.Optimizer):

    ##  Put model into training mode; ensures gradient tracking.

    model.train()
    
    for batch, (X, y) in enumerate(dl):
        
        X, y = X.to(device), y.to(device)

        ##  Run X through model, generate prediction.
        
        y_hat = model(X)

        ##  Calculate error of prediction.

        loss_res = loss(y_hat, y)
        
        ##  Compute gradients numerically via backpropagation, back to
        ##  leaf nodes of graph.
        
        loss_res.backward()

        ##  Iterate parameters.
        
        optimizer.step()

        ##  Reset gradients within graph.
        
        optimizer.zero_grad()

        optimizer.debug_off()

        if batch % 1000 == 0:
                  
            # print(torch.stack([X, y, y_hat], dim=0).T)

            ##  It is not necessary to know the exact parameter
            ##  values, just that they are changing.

            optimizer.debug_off()

            # print(model.state_dict().keys())

            # print(model.state_dict()['linear_relu_stack.0.mu_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.0.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.0.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.0.rho_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.2.mu_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.2.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.2.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.2.rho_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.4.mu_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.4.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.4.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.4.rho_bias'].sum(),
            #       )

            # print(model.state_dict()['linear_relu_stack.0.mu_weight'][0])
            # print(model.state_dict()['linear_relu_stack.0.mu_weight'][0].grad)
            
            # print(torch.round(y_hat), y)

            # print(loss_res.item())
            
            # current = (batch + 1) * len(X)
            
            # print(f"[{current:>5d}/{len(train_dataloader.dataset):>5d}]")

            # print(f"loss_1: {loss_1:>7f}  [{current:>5d}/{size:>5d}]")


def test(dl: DataLoader, model: nn.Module, loss_fn: nn.Module):

    ##  Setup parameters for loss function.

    model.eval()

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        
        for X, y in dl:
            
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dl)
    
    correct /= len(dl.dataset)
    
    print(f"{test_loss:>8f}")


main()
