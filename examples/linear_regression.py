import numpy as np
import pandas as pd
from sqlearn import LinearRegressor

def main():
    train_file = "examples/data/heart_test.tsv"
    test_file = "examples/data/heart_train.tsv"
    train_data = pd.read_csv(train_file, sep="\t", header=0)
    test_data = pd.read_csv(test_file, sep="\t", header=0)

    train_X = train_data.drop("heart_disease", axis=1)
    train_y = train_data["heart_disease"]
    test_X = test_data.drop("heart_disease", axis=1)
    test_y = test_data["heart_disease"]

    epochs = 5
    lr = LinearRegressor(5)
    lr.train(train_X, train_y)

    preds = lr.predict(test_X)

    mse = 0
    for i in range(len(preds)):
        mse += (preds[i] - test_y[i])^2
    mse /= len(preds)

    print(mse)




main()