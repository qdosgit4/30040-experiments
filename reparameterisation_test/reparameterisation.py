#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import PiDataset


train_data = PiDataset("pi_dataset_10000.txt")

train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)


test_data = PiDataset("pi_dataset_2000.txt")

test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)


for batch_data, batch_labels in train_dataloader:

    # print(batch_data)
    print(batch_labels)

    
