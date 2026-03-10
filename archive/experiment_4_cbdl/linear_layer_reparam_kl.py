# mypy: allow-untyped-defs

##  Layer extracted from PyTorch library, edits inspired by
##  bayesian-torch library.

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module


__all__ = [
    "Linear_deterministic",
]


class Linear_reparam_gaussian(Module):
       
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    weight_mu: Tensor
    weight_rho: Tensor
    bias_rho: Tensor

    def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device = None,
            dtype = None,
            w_mu_init: float = 0.25,
            w_rho_init: float = -5,
            b_mu_init: float = 0.25,
            b_rho_init: float = -2.5
    ) -> None:
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()
        
        self.in_features = in_features
        
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.weight_mu = Parameter(torch.empty(out_features,
                                               in_features, **factory_kwargs))
        
        self.weight_rho = Parameter(torch.empty(out_features,
                                                in_features, **factory_kwargs))

        # self.sigma_weight = Parameter(torch.empty(out_features,
        #                                         in_features, **factory_kwargs))

        self.register_buffer('weight_eps', torch.empty(out_features,
                                                        in_features,
                                                        **factory_kwargs),
                                                        persistent=False)

        ##  For calculating KL divergence loss function.

        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        if bias:
                        
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.bias_mu = Parameter(torch.empty(out_features, **factory_kwargs))
        
            self.bias_rho = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.register_buffer('bias_eps',
                                 torch.empty(out_features, in_features, **factory_kwargs),
                                 persistent=False)

            self.register_buffer('prior_bias_mu',
                                 torch.Tensor(out_features),
                                 persistent=False)
            
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)

        else:
            
            self.register_parameter("bias", None, **factory_kwargs)
            
            self.register_parameter("bias_mu", None, **factory_kwargs)
            
            self.register_parameter("bias_rho", None, **factory_kwargs)
            
            self.register_buffer('bias_eps', None, persistent=False)

            self.register_buffer('prior_bias_mu', None, persistent=False)
            
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.reset_parameters(w_mu_init, w_rho_init, b_mu_init, b_rho_init)

        
    def reset_parameters(self, w_mu_init: float, w_rho_init: float, b_mu_init: float, b_rho_init: float) -> None:

        # init.constant_(self.weight_mu, mu_init)
 
        # init.constant_(self.weight_rho, rho_init)
 
        # init.constant_(self.sigma_weight, rho_init)
 
        init.uniform_(self.weight_mu, a = - w_mu_init, b = w_mu_init)
        
        init.uniform_(self.weight_rho, a = w_rho_init, b = w_rho_init)
        
        init.uniform_(self.bias_rho, a = b_rho_init, b = b_rho_init)
        
        # init.uniform_(self.sigma_weight, a = rho_init, b = rho_init)
        
        # init.kaiming_uniform_(self.weight_rho, a=math.sqrt(5))

        ##  Experiment 2 initialised the weight tensor via Kaiming,
        ##  then the bias via the weight tensor.
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            
            # init.constant_(self.bias_mu, mu_init)
        
            # init.constant_(self.bias_rho, rho_init)
        
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            
            init.uniform_(self.bias, -bound, bound)


    def kl_loss(self):
        
        init.uniform_(self.weight_mu, a = - w_mu_init, b = w_mu_init)
        
        init.uniform_(self.weight_rho, a = w_rho_init, b = w_rho_init)
        
        init.uniform_(self.bias_rho, a = b_rho_init, b = b_rho_init)
        
        ##  Find KL divergence between prior distribution and current
        ##  approximation distribution.
        
        kl = self.kl_div(
            self.weight_mu,
            torch.log1p(torch.exp(self.weight_rho)),
            self.prior_weight_mu,
            self.prior_weight_sigma
        )

        ##  Potentially also involve bias parameters in KL divergence
        ##  loss score.
        
        if self.bias_mu is not None:
            
            kl += self.kl_div(
                self.bias_mu,
                torch.log1p(torch.exp(self.bias_rho)),
                self.prior_bias_mu,
                self.prior_bias_sigma
            )
            
        return kl
            

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):

        kl = torch.log(sigma_p) - torch.log(sigma_q) + (
            sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)
                                              ) - 0.5
        
        return kl

            
    def forward(self, input: Tensor, return_kl = True) -> Tensor:

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        self.weight_eps = self.weight_eps.data.normal_(mean = 0, std = 1)

        self.bias_eps = self.bias_eps.data.normal_(mean = 0, std = 1)

        # self.weight_eps = torch.abs(self.weight_eps)
        
        layer_out = F.linear(input,
                        self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * self.weight_eps,
                        self.bias) ### + torch.log1p(torch.exp(self.bias_rho)) * self.bias_eps)

        kl_weight = self.kl_div(self.mu_weight,
                                sigma_weight,
                                self.prior_weight_mu,
                                self.prior_weight_sigma)
            
        if self.mu_bias is not None:

            ##  Generate bias parameter via epsilon.
            
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                
            bias = self.bias_mu + (bias_sigma * self.eps_bias.data.normal_())

            ##  Generate KL loss value related to current bias
            ##  approximation distribution, prior distribution.
            
            kl_bias = self.kl_div(self.bias_mu,
                                  bias_sigma,
                                  self.prior_bias_mu,
                                  self.prior_bias_sigma)

        
        return layer_out, kl_weight, kl_bias

    
    def extra_repr(self) -> str:
        
        """
        Return the extra representation of the module.
        """
        
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
