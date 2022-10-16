import numpy as np
import pandas as pd
import Model

class SGDRegressor(Model):
    """
    A regression model that's trained by stochastic gradient descent
    using a given gradient function.

    Attributes:
    ----------
    calc_gradient(theta, X, y)
        A function to calculate the gradient of the loss function from
        weights (theta), a dataframe of the observations without labels
        (X), and the labels of the observations (y).
    epochs
        The number of passes the model makes through the dataset.
    batch_size
        The number of observations used to calculate the gradient during
        the process of stochastic gradient descent.
    learning_rate
        The rate at which the model changes during stochastic gradient
        descent.
    
    Functions:
    ----------
    train(X, y):
        Trains the model on a dataset and the corresponding labels.
    
    predict(x):
        Makes a prediction from a given observation.

    """

    def __init__(calc_gradient, epochs, batch_size=1, learning_rate=0.001, seed=0):
        """
        Initializes the SGDRegressor.

        Parameters:
        ----------
        calc_gradient(theta, X, y)
            A function to calculate the gradient of the loss function from
            weights (theta), a dataframe of the observations without labels
            (X), and the labels of the observations (y).
        epochs
            The number of passes the model makes through the dataset.
        batch_size
            The number of observations used to calculate the gradient during
            the process of stochastic gradient descent.
        learning_rate
            The rate at which the model changes during stochastic gradient
            descent.
        
        Returns:
        ----------
            An SGDRegressor with initialized hyperparameters. 
        """
        pass

    def train(self, X, y):
        """
        Trains the SGDRegressor using stochastic gradient descent and the
        initialized hyperparameters.

        Parameters:
        ----------
        X
            A dataframe containing rows representing observations 
            without a label column.
        y
            A dataframe (or series) containing the labels for each
            observation in X.
        """
        pass

    def predict(self, X, y):
        """
        Makes a prediction from a given observation.

        Parameters:
        ----------
        x
            A single observation without the label.

        Returns:
        ----------
        y
            The predicted label of the observation.
        """
        pass
