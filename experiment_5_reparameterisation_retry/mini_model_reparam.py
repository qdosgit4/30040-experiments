#!/usr/bin/env python3

import torch
from torch import nn

from linear_bayesian import Linear_bayesian
# from sigmoid_param import Parameterised_sigmoid

class Linear_model(nn.Module):
    
    def __init__(self, n: int):

        ##  n = neurons in 1-n-n-1 network
        
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            Linear_bayesian(1, n),
            nn.ReLU(),
            Linear_bayesian(n, n),
            nn.ReLU(),
            Linear_bayesian(n, 1),
            nn.Sigmoid()
        )

        ##  Non-probabilistic equivalent:

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

        return self.linear_relu_stack(x)
    
