import numpy as np
import pandas as pd
import Model
import random

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
    seed
        The seed used for random processes.
    
    Functions:
    ----------
    train(X, y):
        Trains the model on a dataset and the corresponding labels.
    
    predict(x):
        Makes a prediction from a given observation.

    """

    def __init__(self, calc_gradient, epochs, batch_size=1, learning_rate=0.001, seed=0):
        """
        Initializes the SGDRegressor.

        Parameters:
        ----------
        calc_gradient(theta, X, y)
            A function to calculate the gradient of the loss function from
            weights (theta), a dataframe of a batch of observations without 
            labels (X), and the labels of those observations (y).
        epochs
            The number of passes the model makes through the dataset.
        batch_size
            The number of observations used to calculate the gradient during
            the process of stochastic gradient descent.
        learning_rate
            The rate at which the model changes during stochastic gradient
            descent.
        seed
            The seed used for random processes.
        
        Returns:
        ----------
            An SGDRegressor with initialized hyperparameters. 
        """
        self.calc_gradient = calc_gradient
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        random.seed(self.seed)

    def sgd(self, batch_X, batch_y):
        """
        Performs one step of stochastic gradient descent.

        Parameters:
        ----------
        batch_X
            A dataframe containing a batch of observations.
        batch_y
            A dataframe (or series) containing the labels
            for batch_X
        """
        gradient = self.calc_gradient(self.theta, batch_X, batch_y)
        self.theta = self.theta - self.learning_rate*gradient

    def train(self, X, y):
        """
        Trains the SGDRegressor using stochastic gradient descent and the
        initialized hyperparameters.

        Parameters:
        ----------
        X
            A dataframe containing rows representing observations 
            without a label column. Strictly numeric.
        y
            A dataframe (or series) containing the labels for each
            observation in X.
        """
        ones = np.ones(X.shape[0])
        X_ones = np.hstack((ones, X))
        m = X_ones.shape[0]
        n = X_ones.shape[1]
        self.theta = np.ones(n)
        for epoch in self.epochs:
            for batch in range(m):
                batch_i = random.sample(range(m), self.batch_size)
                batch_X = X_ones.iloc[batch_i]
                batch_y = y.iloc[batch_i]
                self.sgd(batch_X, batch_y)

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
