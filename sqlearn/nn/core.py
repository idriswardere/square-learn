import numpy as np
from .modules import Module
from.modules import LossModule
from .modules import InputLayer
from ..model_interface import Model

class ModuleNode:
    """
    A node of a doubly linked list containing a module.

    Attributes:
    ----------
    module
        The module contained within the node.
    loss
        A boolean representing whether the node is a loss node or not.
    next
        The next ModuleNode in the doubly linked list.
    prev
        The previous ModuleNode in the doubly linked list.
    """
    
    def __init__(self, module: Module, next=None, prev=None) -> None:
        self.module = module
        self.next = next
        self.prev = prev

class NeuralNetwork(Module, Model):
    """
    A modular neural network.

    Attributes:
    ----------
    input_size
        The length of the input vector in the neural network.
    loss_module
        The module used for determining the loss.

    Functions:
    ----------
    add(self, module)
        Add a module to the neural network.
    forward(self, input)
        The output from the forward pass of the neural network given an input.
        Does not include loss module.
    backward(self, dldo)
        The output from the backward pass of the neural network given the
        derivative of the loss with respect to the output of the network
        excluding the loss module. This means that the final added layer
        in the network is considered the "loss".
    backward_with_loss(self, y_hat, y)
        The output from the backward pass of the neural network given the
        derivative of the loss with respect to the output of the network.
        Includes loss module.
    update(self)
        Updates the parameters of the model.
    loss(self, y_hat, y)
        Calculates the loss. Also may update cache in loss module.
    set_loss_module(self, module)
        Sets the loss module.
    train(self, X, Y)
        Trains the neural network from data and its labels.
    predict(self, X)
        Makes predictions given observed data using the neural network.

    
    """
    
    def __init__(self, input_size: float, loss_module: LossModule=None) -> None:
        self.input_size = input_size
        start_module = InputLayer(input_size)
        self.start_node = ModuleNode(start_module)
        self.end_node = self.start_node
        self.loss_module = loss_module
    
    def add(self, module: Module) -> None:
        """
        Add a module to the neural network.

        Parameters:
        -----------
        module
            The module to be added.
        """
        node = ModuleNode(module, prev=self.end_node)
        self.end_node.next = node
        self.end_node = node

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        The output from the forward pass of the neural network given an input.
        Does not include loss module.

        Parameters:
        ----------
        input
            The input data to perform the forward pass with.

        Returns:
        ----------
        forward_pass
            The output of the forward pass.
        """
        node = self.start_node
        forward_pass = input
        while node != None:
            forward_pass = node.module.forward(forward_pass)
            node = node.next
        return forward_pass # y_hat

    def backward(self, dldo: np.ndarray) -> np.ndarray:
        """
        The output from the backward pass of the neural network given the
        derivative of the loss with respect to the output of the network
        excluding the loss module. This means that the final added layer
        in the network is considered the "loss".

        Parameters:
        ----------
        dldo
            The derivative of the loss with respect to the output of the
            network.
        
        Returns:
        ----------
        dldi
            The deriviative of the loss with respect to the input of the 
            network.
        """
        backward_pass = dldo
        node = self.end_node
        while node != None:
            backward_pass = node.module.backward(backward_pass)
            node = node.prev
        return backward_pass
    
    def backward_with_loss(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        The output from the backward pass of the neural network given the
        derivative of the loss with respect to the output of the network.
        Includes loss module.

        Parameters:
        ----------
        y_hat
            The output from a forward pass of the model.
        y
            The expected output.
        Returns:
        ----------
        dldi
            The deriviative of the loss with respect to the input of the 
            network.
        loss
            The calculated loss from y_hat and y.
        """
        loss = self.loss_module.forward(y_hat, y)
        backward_pass = self.loss_module.backward()
        node = self.end_node
        while node != None:
            backward_pass = node.module.backward(backward_pass)
            node = node.prev
        return backward_pass, loss

    def update(self) -> None:
        """
        Updates the parameters of the model.
        """
        node = self.start_node
        while node != None:
            node.module.update()
            node = node.next

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the loss. Also may update cache in loss module.

        Parameters:
        ----------
        y_hat
            The output from a forward pass of the model.
        y
            The expected output.

        Returns:
        ----------
        loss
            The calculated loss.
        """
        result = self.loss_module.forward(y_hat)
        return result
    
    def set_loss_module(self, loss_module: LossModule) -> None:
        """
        Sets the loss module of the neural network.

        Parameters:
        ----------
        module
            The loss module to be set.
        """
        self.loss_module = loss_module

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int=100, seed: int=0) -> float: # TODO
        """
        Trains the neural network from data and its labels.

        Parameters:
        ----------
        X
            The observation data.
        Y
            The true labels of the observation data.

        Returns:
        ----------
        losses
            An array containing the average loss at the end of each epoch.
        """
        np.random.seed(seed)
        losses = np.zeros(epochs)
        for epoch in range(epochs):
            np.random.shuffle(X)
            for i, x in enumerate(X):
                y_hat = self.forward(x)
                y = Y[i]
                dldi, loss = self.backward_with_loss(y_hat, y)
                losses[epoch] += loss
                self.update()
            losses[epoch] /= X.shape[0]
        return losses





    def predict(self, X: np.ndarray) -> np.ndarray: # TODO
        """
        Makes predictions given observed data using the neural network.

        Parameters:
        ----------
        X
            The observation data.
        """
        pass
    
