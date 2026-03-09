#!/usr/bin/env python3

import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
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

    ##  Load up data.

    batch_size_pi = args.batch_n

    train_data = Pi_dataset("pi_dataset_10000.txt")

    train_dl = DataLoader(train_data, batch_size = batch_size_pi,
                                  shuffle=True)

    test_data = Pi_dataset("pi_dataset_2000.txt")

    test_dl = DataLoader(test_data, batch_size = batch_size_pi,
                                 shuffle=True)

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

    ##  Either load weights or skip.

    print(args.weights_load)

    if args.weights_load is not None:
        
        state_dict = torch.load(args.weights_load, weights_only=True)

    else:

        print("Training initialised.")

        run_training_loop(model, train_dl, test_dl, 1)

    ##  Generate timestamp and store weights for later loading back.
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"model_weights_{args.weights_name.replace(' ', '-')}_{timestamp}.pth"
        
    torch.save(model.state_dict(), filename)
    

main()
