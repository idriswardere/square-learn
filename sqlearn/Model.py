import numpy as np
import pandas as pd

class Model:
    """
    An interface for machine learning models.

    Functions:
    ----------
    train(X, y):
        Trains the model on a dataset and the corresponding labels.
    
    predict(x):
        Makes a prediction from a given observation.
    """ 

    def train(self, X, y):
        """
        Trains the model on a dataset and the corresponding labels.

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

    def predict(self, x):
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



