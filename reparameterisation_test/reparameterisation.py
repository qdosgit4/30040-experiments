#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import PiDataset


##  Load up data.

batch_size_pi = 10


train_data = PiDataset("pi_dataset_10000.txt")

train_dataloader = DataLoader(train_data, batch_size=batch_size_pi,
                              shuffle=True)


test_data = PiDataset("pi_dataset_2000.txt")

test_dataloader = DataLoader(test_data, batch_size=batch_size_pi, shuffle=True)


for batch_data, batch_labels in train_dataloader:

    # print(batch_data)
    print(batch_labels)



v = torch.tensor([1],
                 dtype = torch.bfloat16,
                 requires_gradient = True)

##  Start off mu at pi to speed up convergence.

mu = torch.tensor(3.14159,
                  dtype = torch.bfloat16,
                  requires_gradient = True)


##  Extract data using mask.

batch_data_0 = batch_data[batch_labels == 0]

batch_data_1 = batch_data[batch_labels == 1]



loss_0 = nn.L1Loss()

loss_1 = nn.GaussianLoss()


mu_n_0 = mu.repeat(batch_data_0.shape[0])

mu_n_1 = mu.repeat(batch_data_1.shape[0])

v_n_1 = mu.repeat(batch_data_1.shape[0])


output = loss_0(mu_n_0, _data_0) + loss_1(mu_n_1, _data_1, v_n_1)

output.backward()
