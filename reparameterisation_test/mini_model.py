#!/usr/bin/env/python3

import torch
from torch import nn


class Linear_model(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1)
        )
        

    def forward(self, x):
        
        logits = self.linear_relu_stack(x)
        
        return logits
