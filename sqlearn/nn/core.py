import numpy as np
from .modules import Module

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
        self.loss = loss # might scrap for self.loss_module in NN?
        self.next = next
        self.prev = prev

class NeuralNetwork(Module): # also implements model?? (would need train/predict functions)
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

    def add(self, module): # TODO
        pass
    
    def set_loss(self, module): # TODO
        pass
    
