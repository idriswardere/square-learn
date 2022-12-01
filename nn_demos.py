import numpy as np
from sqlearn.nn.modules import Linear
from sqlearn.nn.modules import Sigmoid
from sqlearn.nn.modules import MeanSquaredError
from sqlearn.nn import NeuralNetwork



def main():
    # Processing the example data (pandas)

    train_file = "examples/data/heart_test.tsv"
    test_file = "examples/data/heart_train.tsv"
    train_data = np.loadtxt(train_file, skiprows=1)
    test_data = np.loadtxt(test_file, skiprows=1)

    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    # #Testing the modules

    # lin1 = Linear(train_X.shape[1], 1)
    # sig = Sigmoid()
    # lf1 = lin1.forward(train_X[0])
    # sf1 = sig.forward(lf1)
    # sb1 = sig.backward(1)
    # lb1 = lin1.backward(sb1)
    # print(train_X[0])
    # print("Lin1 Forward:", lf1)
    # print("Sig Forward:", sf1)
    # #print("Lin1 Weights:", lin1.weights)
    # print("--------------")
    # print("Sig Backward:", sb1)
    # print("Lin1 Backward:", lb1)

    # Testing NeuralNetwork

    input_size = train_X.shape[1]
    nn = NeuralNetwork(input_size)
    nn.add(Linear(input_size, 1))
    nn.add(Sigmoid())
    nn.set_loss_module(MeanSquaredError())
    nnf1 = nn.forward(train_X[0])
    print(f"Forward: {nnf1}")
    print(f"Backward: {nn.backward(1)}")
    print(f"Backward with loss: {nn.backward_with_loss(nnf1, train_y[0])}")

    losses = nn.train(train_X, train_y, epochs=300)
    print(losses)
    preds = nn.predict(test_X)
    print(preds)
    print(test_y)


main()