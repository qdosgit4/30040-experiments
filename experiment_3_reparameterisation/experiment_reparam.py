#!/usr/bin/env python3

import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset
from mini_model_reparam import Linear_model

from experiment_training_lib import run_training_loop

##  Initialisation.

parser = argparse.ArgumentParser()

parser.add_argument('--neurons-n',
                    required = True,
                    type = int,
                    help = 'Quantity of n neurons in 1-n-n-1 network. 512, or 1024 recommended.')

parser.add_argument('--batch-n',
                    required = True,
                    type = int,
                    help = 'Size of batch from dataset during training/testing. 256 recommended.')

parser.add_argument('--training-epochs',
                    required = True,
                    type = int,
                    help = 'Quantity of training epochs to run.')

parser.add_argument('--weights-name',
                    required = False,
                    type = str,
                    help = 'Name to append to stored weights filename.')

parser.add_argument('--weights-load',
                    required = False,
                    type = str,
                    help = 'Name to append to stored weights filename.')

parser.add_argument('--sigmoid-k',
                    required = False,
                    default = 1,
                    type = float,
                    help = 'Initial Sigmoid parameter to use as likelihood model.')
parser.add_argument('--mu-init',
                    required = False,
                    default = 0.25,
                    type = float,
                    help = 'Initial value of mu to set mu of probabilitiy distributions of weights to.')
parser.add_argument('--rho-init',
                    required = False,
                    default = -5,
                    type = float,
                    help = 'Initial value of rho to set mu of probabilitiy distributions of weights to.')

args = parser.parse_args()

for key, value in vars(args).items():
            print(f"{key}: {value}")

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(239852)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def main():

    ##  Define model.

    model = Linear_model(
                args.neurons_n,
                float(args.sigmoid_k),
                float(args.mu_init),
                float(args.rho_init)
    ).to(device)

    ##  Ensure multiple GPUs used if available.
    
    if torch.cuda.device_count() > 1:
        
        print(f"Using {torch.cuda.device_count()} GPUs!")
        
        model = nn.DataParallel(model, device_ids=[0, 1])

    ##  Either load weights or load training data.

    if args.weights_load is not None:
        
        run_utilisation_loop(model, args.weights_load)

    else:

        ##  Load up data.

        batch_size_pi = args.batch_n

        train_dl = DataLoader(
                    Pi_dataset("pi_dataset_10000.txt"),
                    batch_size = args.batch_n,
                    shuffle=True
        )

        test_dl = DataLoader(
                    Pi_dataset("pi_dataset_2000.txt"),
                    batch_size = args.batch_n,
                    shuffle=True
        )

        filename = f"model_weights_{args.weights_name.replace(' ', '-')}_batch_{args.batch_n}_epochs_{args.training_epochs}_{timestamp}.pth"

        run_training_loop(model, train_dl, test_dl, args.training_epochs, filename)
    

def run_utilisation_loop(model: nn.Module, weights_path: str):

    ##  Load weights.

    checkpoint = torch.load(weights_path,
                           weights_only = True,
                           map_location=torch.device('cpu')
                           )

    old_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    ##  Fix naming scheme and load module.

    model.load_state_dict(
                {k.replace('module.', '', 1): v for k, v in old_state_dict.items()}
    )

    res = []
    
    interval = 1 / (2 ** 6)

    dl = DataLoader(
                TensorDataset(
                    torch.arange(1.57, 4.71 + interval, interval)
                ),
                batch_size = 1,
                shuffle = False
    )
    
    outputs = []

    with torch.no_grad():

        all_tensors = []

        for i in range(2):

            outputs = []
            
            for batch, (X,) in enumerate(dl):

                X = X.to(device)

                # print(X)

                ##  Run X through model, generate prediction.

                y_hat = model(X)

                # print(y_hat)

                outputs.append(y_hat.squeeze(0))

            inner_tensor = torch.stack(inner_list, dim=0)

            all_tensors.append(inner_tensor)

        final_tensor = torch.stack(all_outputs, dim=0)

        print(final_tensor)

            

    
main()

