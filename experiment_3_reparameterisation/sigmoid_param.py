#!/usr/bin/env python3

import torch
import torch.nn as nn

class Parameterised_sigmoid(nn.Module):

    def __init__(self, k_init = 75):
        
        super().__init__()
        
        self.k = nn.Parameter(torch.tensor([k_init]))

        
    def forward(self, x):
        
        return 1 / (1 + torch.exp(-self.k * x))

