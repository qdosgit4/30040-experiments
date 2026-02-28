# mypy: allow-untyped-defs

##  Layer extracted from PyTorch library.

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


class Linear_deterministic(Module):
       
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    mu_weight: Tensor
    rho_weight: Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device = None,
            dtype = None,
            mu_init: float = 0.25,
            rho_init: float = 0.25
    ) -> None:
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()
        
        self.in_features = in_features
        
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.mu_weight = Parameter(torch.empty(out_features,
                                               in_features, **factory_kwargs))
        
        self.rho_weight = Parameter(torch.empty(out_features,
                                                in_features, **factory_kwargs))

        self.sigma_weight = Parameter(torch.empty(out_features,
                                                in_features, **factory_kwargs))

        self.register_buffer('eps_weight', torch.empty(out_features,
                                                        in_features,
                                                        **factory_kwargs),
                                                        persistent=False)

        if bias:
                        
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.mu_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        
            self.rho_bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.sigma_bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.register_buffer('eps_weight',
                             torch.empty(out_features, in_features, **factory_kwargs),
                                 persistent=False)

        else:
            
            self.register_parameter("bias", None, **factory_kwargs)
            
            self.register_parameter("mu_bias", None, **factory_kwargs)
            
            self.register_parameter("rho_bias", None, **factory_kwargs)
            
            self.register_parameter("sigma_bias", None, **factory_kwargs)
            
            self.register_buffer('eps_weight', None, persistent=False)

        self.reset_parameters(mu_init, rho_init)

        
    def reset_parameters(self, mu_init: float, rho_init: float) -> None:
        
        init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        
        init.kaiming_uniform_(self.rho_weight, a=math.sqrt(5))
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            
            init.uniform_(self.bias, -bound, bound)

            
    def forward(self, input: Tensor) -> Tensor:
        
        # self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        # self.sigma_weight = torch.exp(self.rho_weight)

        # input_identity = torch.ones_like(input)

        # self.weight = self.mu_weight + self.rho_weight

        # print(self.rho_weight)

        # print(torch.exp(self.rho_weight))
        
        return F.linear(input,
                        torch.log1p(self.rho_weight),
                        self.bias)

    
    def extra_repr(self) -> str:
        
        """
        Return the extra representation of the module.
        """
        
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
