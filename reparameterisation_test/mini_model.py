#!/usr/bin/env/python3

import torch
from torch import nn


class Linear_model(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        v_loss = nn.Parameter(torch.Tensor([1]))

        mu_loss = nn.Parameter(torch.Tensor([3.14159]))

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            # nn.Linear(1, 3),
            # nn.ReLU(),
            # nn.Linear(3, 1),
        )
        

    def forward(self, x):
        
        logits = self.linear_relu_stack(x)

        out = torch.sigmoid(logits)
        
        return out
