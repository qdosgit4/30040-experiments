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


torch.set_default_tensor_type(torch.bfloat16)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


####  Load up data.

batch_size_pi = 10

train_data = Pi_dtaset("pi_dataset_10000.txt")

train_dataloader = DataLoader(train_data, batch_size=batch_size_pi,
                              shuffle=True)

test_data = Pi_dtaset("pi_dataset_2000.txt")

test_dataloader = DataLoader(test_data, batch_size=batch_size_pi, shuffle=True)


v_w1 = nn.Parameter([1])
    
mu_w1 = nn.Parameter([0])

normal_dist = distributions.Normal(0, 1)
    
w = mu_w1 + torch.sqrt(v_w1) * normal_dist.sample((1, 1))

c = nn.Parameter([0.5])

v_loss = nn.Parameter([1])

####  Start off mu at pi to speed up convergence.

mu_loss = nn.Parameter(3.14159)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


##  Training.

for batch_data, batch_labels in train_dataloader:

    # print(batch_data)
    print(batch_labels)


####  Extract data using mask.

batch_data_0 = batch_data[batch_labels == 0]

batch_data_1 = batch_data[batch_labels == 1]


model = Linear_model().to(device)

print(model[0].weight.data)

model[0].weight.data = w

model[0].bias.data = c


loss_0 = nn.L1Loss()

loss_1 = nn.GaussianLoss()


mu_l_0 = mu_loss.repeat(batch_data_0.shape[0])

mu_l_1 = mu_loss.repeat(batch_data_1.shape[0])

v_l_1 = v_loss.repeat(batch_data_1.shape[0])


output = loss_0(mu_l_0, _data_0) + loss_1(mu_l_1, _data_1, v_l_1)

output.backward()
