#!/usr/bin/env/python3

import torch
from torch import nn

from linear_reparam_layer import Linear_gaussian_reparam


class Linear_model(nn.Module):
    
    def __init__(self, n: int):
        
        super().__init__()
        
        v_loss = nn.Parameter(torch.Tensor([1]))

        mu_loss = nn.Parameter(torch.Tensor([3.14159]))

        self.linear_relu_stack = nn.Sequential(
            Linear_gaussian_reparam(1, n),
            nn.ReLU(),
            Linear_gaussian_reparam(n, n),
            nn.ReLU(),
            Linear_gaussian_reparam(n, 1)
        )

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(1, n),
        #     nn.ReLU(),
        #     nn.Linear(n, n),
        #     nn.ReLU(),
        #     nn.Linear(n, 1)
        # )

        # nn.init.uniform_(self.linear_relu_stack[0].weight, a=-0.25, b=0.25)
        # nn.init.uniform_(self.linear_relu_stack[2].weight, a=-0.25, b=0.25)
        # nn.init.uniform_(self.linear_relu_stack[4].weight, a=-0.25, b=0.25)
        

    def forward(self, x: torch.Tensor):
        
        logits = self.linear_relu_stack(x)

        out = torch.sigmoid(logits)
        
        return out
