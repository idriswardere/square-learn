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
        Returns the output of the module's forward pass

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


class ModuleNode:
    """
    A node of a doubly linked list containing a module.

    Parameters:
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
    
    def __init__(self, module, loss=False, next=None, prev=None):
        self.module = module
        self.loss = loss
        self.next = next
        self.prev = prev

class NeuralNetwork(Module):
    """
    A neural network.
    """
    
    def __init__(self):
        pass
    
    def forward(self, input): # TODO
        pass

    def backward(self, dldo): # TODO
        pass

    def update(self): # TODO
        pass

    def add(): # TODO
        pass
    
