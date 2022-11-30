import numpy as np
from .neural_network_core import Module

class InputLayer(Module):
    """
    The input layer of a neural network.

    Attributes:
    ----------
    input_size
        The size of the input data.
    """

    def __init__(self, input_size: int):
        """
        Initializes the input layer.

        Parameters:
        ----------
        input_size
            The size of the input data.
        """
        self.input_size = input_size
        self.output_size = input_size
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        return input
    
    def backward(self, dldo: np.ndarray) -> np.ndarray:
        return dldo
        

class Sigmoid(Module):
    """
    A sigmoid module. Performs the sigmoid operation elementwise on the input.
    """

    def __init__(self):
        self.o_cache = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of sigmoid module.
        """
        exp = np.exp(input)
        output = exp/(1+exp)
        self.o_cache = output
        return output
    
    def backward(self, dldo: np.ndarray) -> np.ndarray:
        """
        Backward pass of sigmoid module.
        """
        output = self.o_cache
        dldi = output*(1-output)*dldo
        return dldi


def random_init(shape: tuple, scale=1) -> np.ndarray:
    random = (np.random.random(shape) - 0.5)*scale
    return random

class Linear(Module):
    """
    A linear module. Fully connected linear layer using random weight
    initialization.

    Attributes:
    ----------
    input_size
        The number of nodes outputted by the previous module.
    output_size
        The number of nodes outputted by this module. Also the number of hidden
        nodes.
    learning_rate (default=0.01)
        The rate that the weights are modified by the gradient.
    r_scale (default=1)
        The scaling of the random initialization of the weights. At its default,
        the weights are randomly initialized in the range (-0.5, 0.5)*r_scale.
    """

    def __init__(self, input_size: int, output_size: int, learning_rate=0.01, r_scale=1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = random_init((output_size, 1+input_size), scale=r_scale)
    
    def forward(self, input: np.ndarray):
        """
        Forward pass of linear module.
        """
        one = np.ones(1)
        input_b = np.hstack((one, input)) # bias term folded into input
        print(f"Weights {self.weights.shape} Input_b {input_b.shape}")
        output = self.weights @ input_b # keepdims?
        self.input_b_cache = input_b
        return output

    def backward(self, dldo: np.ndarray):
        """
        Backward pass of linear module.
        """
        dldi = self.weights[:, 1:].T @ dldo
        self.dldw = np.outer(dldo, self.input_b_cache)
        return dldi

    def update(self):
        """
        Updating the weight matrix.
        """
        self.weights = self.weights - self.learning_rate*self.dldw


# TODO: Loss modules



    