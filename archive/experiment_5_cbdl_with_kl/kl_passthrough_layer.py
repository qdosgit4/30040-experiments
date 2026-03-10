#!/usr/bin/env python3

##  This captures the KL loss function part related to the difference
##  between the current multidimensional approximation function and
##  the prior distribution.

##  Example usage, within an nn.Sequential:
##  KL_passthrough_wrapper(nn.ReLU(x))
##  KL_passthrough_wrapper(Linear_layer_gaussian(x)) 


class KL_passthrough_wrapper(nn.Module):
    
    def __init__(self, layer):
        
        super().__init__()
        
        self.layer = layer

        
    def forward(self, inputs):

        ##  Unpack the tuple containing KL data and previous layer
        ##  output.
        
        x, kl_data = inputs

        ##  Forward data through network, whilst passing on KL loss
        ##  data.
        
        return self.layer(x), kl_data

    
class KL_recalculate_wrapper(nn.Module):
    
    def __init__(self, layer):
        
        super().__init__()
        
        self.layer = layer

        
    def forward(self, inputs):

        ##  Unpack the tuple containing KL data and previous layer
        ##  output.
        
        x, kl_data = inputs

        x_new, kl_data_new = self.layer(x)

        ##  Forward data through network, whilst passing on KL loss
        ##  data.

        ##  Concatenate the KL data.

        ##  TODO

        ##  Return output of linear layer and all KL data.
                
        return x_new, kl_data_concat
