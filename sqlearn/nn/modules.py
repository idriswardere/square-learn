import numpy as np

class Module:
    """
    An interface for modules in the modular neural network.

    Functions:
    ----------
    forward(self, input):
        Returns the output of the module's forward pass, where the input is
        either the input data or a previous module's forward pass.
    backward(self, dldo):
        Given the derivative of the loss with respect to the output of the 
        module, dl/do, this returns the derivative of the loss with respect to the 
        input of the module, dl/di.
    update(self):
        Updates the parameters of the model. Depending on the module, this may do
        nothing.
    """

    def forward(self, input):
        """
        Returns the output of the module's forward pass.

        Parameters:
        ----------
        input
            The input data or a previous module's forward pass.

        Returns:
        ----------
        output
            The output of the module's forward pass.
        """
        pass

    def backward(self, dldo):
        """
        Returns the derivative of the loss with respect to the 
        input of the module.

        Parameters:
        ----------
        dldo
            The derivative of the loss with respect to the output of the module.

        Returns:
        ----------
        dldi
            The derivative of the loss with respect to the input of the module.
        """
        pass

    def update(self):
        """
        Updates the parameters of the module if necessary.
        """
        pass


class InputLayer(Module):
    """
    The input layer of a neural network.

    Attributes:
    ----------
    input_size
        The size of the input data.
    """

    def __init__(self, input_size: int) -> None:
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


def random_init(shape: tuple, scale: float=1) -> np.ndarray:
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

    def __init__(self, input_size: int, output_size: int,
                learning_rate: float=0.01, r_scale: float=1,
                seed: int=None) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        if seed:
            np.random.seed(seed)
        self.weights = random_init((output_size, 1+input_size), scale=r_scale)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of linear module.
        """
        one = np.ones(1)
        input_b = np.hstack((one, input)) # bias term folded into input
        output = self.weights @ input_b # keepdims?
        self.input_b_cache = input_b
        return output

    def backward(self, dldo: np.ndarray) -> np.ndarray:
        """
        Backward pass of linear module.
        """
        #print(f"weights: {self.weights.T.shape} dldo dim: {np.ndim(dldo)}")
        #print(dldo)
        dldi = self.weights[:, 1:].T @ dldo
        self.dldw = np.outer(dldo, self.input_b_cache)
        return dldi

    def update(self) -> None:
        """
        Updating the weight matrix.
        """
        self.weights = self.weights - self.learning_rate*self.dldw
        

class Sigmoid(Module):
    """
    A sigmoid module. Performs the sigmoid operation elementwise on the input.
    """

    def __init__(self) -> None:
        self.output_cache = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of sigmoid module.
        """
        exp = np.exp(input)
        output = exp/(1+exp)
        self.output_cache = output
        return output
    
    def backward(self, dldo: np.ndarray) -> np.ndarray:
        """
        Backward pass of sigmoid module.
        """
        output = self.output_cache
        dldi = output*(1-output)*dldo
        return dldi


# TODO: Relu/elu & other modules (convolution?)


class LossModule:
    """
    An interface for loss modules in the modular neural network. These modules
    represent loss functions.

    Functions:
    ----------
    forward(self, input):
        Returns the output of the module's forward pass, where the input is
        either the input data or a previous module's forward pass.
    backward(self, dldo):
        Given the derivative of the loss with respect to the output of the 
        module, dl/do, this returns the derivative of the loss with respect to the 
        input of the module, dl/di.
    update(self):
        Updates the parameters of the model. Depending on the module, this may do
        nothing.
    """

    def forward(self, y_hat, y):
        """
        Returns the output of the module's forward pass.

        Parameters:
        ----------
        input
            The output of a model's forward pass.

        Returns:
        ----------
        output
            The output of the loss function.
        """
        pass

    def backward(self):
        """
        Returns the derivative of the loss with respect to the 
        input of the module.

        Returns:
        ----------
        dldi
            The derivative of the loss with respect to the input of the module.
        """
        pass


class MeanSquaredError(LossModule):
    """
    The mean squared error module. This module represents a loss function.
    """
    
    def __init__(self) -> None:
        self.y_hat_cache = None
        self.y_cache = None

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Forward pass of MSE. Calculates the mean squared error.
        """
        loss = np.sum(np.square(y_hat - y))
        loss /= y_hat.shape[0]
        self.y_hat_cache = y_hat
        self.y_cache = y
        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass of MSE.
        """
        y_hat = self.y_hat_cache
        y = self.y_cache
        if np.ndim(y) == 0: # turns floats to arrays to match output
            y = np.array([y])
        assert(y_hat.shape == y.shape)
        n = y_hat.shape[0]
        backward_pass = (2/n)*(y_hat-y)
        return backward_pass

# TODO: Cross entropy loss

    