import numpy as np
import pandas as pd
from sqlearn.LinearRegressor import LinearRegressor

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
    lr = LinearRegressor(epochs, batch_size=2)
    lr.train(train_X, train_y)

    preds = lr.predict(test_X)
    preds = preds > 0.5
    print(preds)

    error_count = 0
    for i in range(len(preds)):
        if preds.loc[i] != test_y.loc[i]:
            error_count += 1
    error_rate = error_count/len(preds)
    print(error_rate)




main()