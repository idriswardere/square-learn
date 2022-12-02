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
    output_size = 1
    lr = 0.0001

    nn = NeuralNetwork(input_size, loss_module=MeanSquaredError())
    nn.add(Linear(input_size, 4, seed=5, learning_rate=lr))
    nn.add(Linear(4, 3, learning_rate=lr))
    nn.add(Linear(3, output_size, learning_rate=lr))
    nn.add(Sigmoid())

    nnf1 = nn.forward(train_X[0])
    print(f"Forward: {nnf1}")
    print(f"Backward: {nn.backward(np.ones(output_size))}")
    print(f"Backward with loss: {nn.backward_with_loss(nnf1, test_y[0])}")

    # train_y2_col = np.reshape(train_y, (train_y.shape[0], 1))
    # train_y2 = np.hstack((train_y2_col, train_y2_col))
    # test_y2_col = np.reshape(test_y, (test_y.shape[0], 1))
    # test_y2 = np.hstack((test_y2_col, test_y2_col))

    losses = nn.train(train_X, train_y, epochs=200)
    print("First epoch loss vs. Last epoch loss: ", losses[0], losses[-1])
    preds = nn.predict(test_X)
    preds_vec = [pred[0] > 0.5 for pred in preds]
    num_errors = np.sum(np.abs(test_y-preds_vec))
    error_rate = num_errors/len(test_y)
    print("Error rate:", error_rate)
    #print("First few test_y and pred differences", (test_y2-preds))


main()