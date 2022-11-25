import numpy as np
import pandas as pd
from neural_network_core import Module

class InputLayer(Module):
    """
    The input layer of a neural network. This layer will usually be first.

    Attributes:
    ----------
    input_size
        The size of the input data.
    """

    def __init__(self, input_size) -> None:
        """
        Initializes the input layer.

        Parameters:
        ----------
        input_size
            The size of the input data.
        """
        self.input_size = input_size
        self.output_size = input_size
    
    def forward(self, input):
        return input
    
    def backward(self, dldo):
        return dldo
        

class Sigmoid(Module):
    """
    A sigmoid module. Performs the sigmoid operation elementwise on the input.
    """

    def __init__(self):
        self.o_cache = None

    def forward(self, input):
        """
        Forward pass of sigmoid module.
        """
        exp = np.exp(input)
        output = exp/(1+exp)
        self.o_cache = output
        return output
    
    def backward(self, dldo):
        """
        Backward pass of sigmoid module.
        """
        output = self.o_cache
        dldi = output*(1-output)*dldo
        return dldi


    