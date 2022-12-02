import numpy as np
from sqlearn.nn.modules import Linear
from sqlearn.nn.modules import Sigmoid
from sqlearn.nn.modules import ReLU
from sqlearn.nn.modules import MeanSquaredError
from sqlearn.nn import NeuralNetwork



def main():

    # Processing the example data (numpy)

    train_file = "examples/data/heart_test.tsv"
    test_file = "examples/data/heart_train.tsv"
    train_data = np.loadtxt(train_file, skiprows=1)
    test_data = np.loadtxt(test_file, skiprows=1)

    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    # Testing NeuralNetwork

    input_size = train_X.shape[1]
    output_size = 1
    lr = 0.01
    np.random.seed(0)

    nn = NeuralNetwork(input_size, loss_module=MeanSquaredError())
    nn.add(Linear(input_size, 7, learning_rate=lr))
    nn.add(ReLU())
    nn.add(Linear(7, output_size, learning_rate=lr))
    nn.add(Sigmoid())
    
    losses = nn.train(train_X, train_y, epochs=50)
    print("First epoch loss vs. Last epoch loss: ", losses[0], losses[-1])
    preds = nn.predict(train_X)
    preds_vec = [pred[0] > 0.5 for pred in preds]
    num_errors = np.sum(train_y != preds_vec)
    error_rate = num_errors/len(train_y)
    print("Error rate:", error_rate)


main()