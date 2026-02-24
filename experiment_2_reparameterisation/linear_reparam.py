#!/usr/bin/env/python3

##  Extracted from bayesian-torch library.

import torch
from torch import nn


class Linear_reparam(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        
        """
        Implements Linear layer with reparameterization trick.

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__()

        ##  Standard linear layer features.

        self.in_features = in_features
        
        self.out_features = out_features

        ##  The prior distributions of the weights.
        
        self.prior_mean = prior_mean
        
        self.prior_variance = prior_variance        

        ##  The initial distribution parameters of the distributions
        ##  to be trained.
        
        ##  This would usually be equivalent to the prior, but this allows a
        ##  pretrained drop-in.
        
        self.posterior_mu_init = posterior_mu_init  # mean of weight
        
        # variance of weight --> sigma = log (1 + exp(rho))
        
        self.posterior_rho_init = posterior_rho_init
        
        self.bias = bias

        ##  Setup training distribution parameters.

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))

        ##  Register buffer: "data that is not a model parameter but
        ##  that is part of the module's state". E.g. during batch
        ##  normalisation.

        ##  eps_weight is used to generate weights during forward
        ##  propagation.
        
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        ##  Prior weight mu/sigma are used to calculate KL divergence.
        
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        if bias:
            
            self.mu_bias = Parameter(torch.Tensor(out_features))
            
            self.rho_bias = Parameter(torch.Tensor(out_features))
            
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
            
        else:
            
            self.register_buffer('prior_bias_mu', None, persistent=False)
            
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            
            self.register_parameter('mu_bias', None)
            
            self.register_parameter('rho_bias', None)
            
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()
        # self.quant_prepare=False

        
    # def prepare(self):

    #     ##  Quantization regards converting continuous variables to a
    #     ##  set of discrete variables. This introduces some quantities
    #     ##  of errors. Qconfig describes how to do so. Minmax involves
    #     ##  taking a tensor and extracting elements.

    #     self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
    #                                      QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
    #     self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
    #                                      QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        
    #     self.dequant = torch.quantization.DeQuantStub()
    #     self.quant_prepare=True

        
    def init_parameters(self):
        
        self.prior_weight_mu.fill_(self.prior_mean)
        
        self.prior_weight_sigma.fill_(self.prior_variance)
        

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

        
        if self.mu_bias is not None:
            
            self.prior_bias_mu.fill_(self.prior_mean)
            
            self.prior_bias_sigma.fill_(self.prior_variance)
            
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)


    ##  Find natural logarithm of exponent of rho weight.

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        ##  Find KL divergence between prior distribution and current
        ##  distribution.
        
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)

        ##  Potentially also involve bias parameters in KL divergence
        ##  loss score.
        
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    
    def forward(self, input, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False
            
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        ##  Generate epsilon weight via normal distribution - see KL
        ##  loss function, original Bayes-by-backprop paper.
        
        eps_weight = self.eps_weight.data.normal_()

        ##  Generate weight via sigma and epsilon weights.
        
        tmp_result = sigma_weight * eps_weight
        
        weight = self.mu_weight + tmp_result

        if return_kl:

            ##  Find difference between newly generated approximation
            ##  weight function and prior weight function.
            
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.mu_bias is not None:

            ##  Generate bias parameter via epsilon.
            
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())

            ##  Generate KL loss value related to current bias
            ##  approximation distribution, prior distribution.

            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        ##  Run input through standard linear layer.

        out = F.linear(input, weight, bias)

        # if self.quant_prepare:
        #     # quint8 quantstub
        #     input = self.quint_quant[0](input) # input
        #     out = self.quint_quant[1](out) # output

        #     # qint8 quantstub
        #     sigma_weight = self.qint_quant[0](sigma_weight) # weight
        #     mu_weight = self.qint_quant[1](self.mu_weight) # weight
        #     eps_weight = self.qint_quant[2](eps_weight) # random variable
        #     tmp_result =self.qint_quant[3](tmp_result) # multiply activation
        #     weight = self.qint_quant[4](weight) # add activatation

        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out
