#!/usr/bin/env/python3

import torch
from torch import nn

from linear_layer_reparam_v2 import Linear_reparam_gaussian


class Linear_model(nn.Module):
    
    def __init__(self, n: int, l_mu: float, l_b: float, w_mu: float, w_rho: float):
        
        super().__init__()

        self.laplace_mu = l_mu

        self.laplace_b = l_b

        self.linear_relu_stack = nn.Sequential(
            Linear_reparam_gaussian(1, n, w_mu_init = w_mu, w_rho_init = w_rho),
            nn.ReLU(),
            Linear_reparam_gaussian(n, n),
            nn.ReLU(),
            Linear_reparam_gaussian(n, 1),
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

        y_hat = self.linear_relu_stack(x)

        if y_hat < self.laplace_mu:

            return 0.5 * exp((y_hat - self.laplace_mu) / self.laplace_b)

        else:

            return 1 - 0.5 * exp(-(y_hat - self.laplace_mu) / self.laplace_b)

        # lpdf_y_hat = torch.tensor(
        #     1 / (2 * self.laplace_b)) * torch.exp(
        #         -torch.div(
        #             torch.abs(y_hat - self.laplace_mu),
        #             self.laplace_b)
        #     )

        # return lpdf_y_hat

