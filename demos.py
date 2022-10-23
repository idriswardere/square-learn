import numpy as np
import pandas as pd
from sqlearn import LinearRegressor
from sqlearn import LogisticRegressor

def main():
    # Processing the example data

    train_file = "examples/data/heart_test.tsv"
    test_file = "examples/data/heart_train.tsv"
    train_data = pd.read_csv(train_file, sep="\t", header=0)
    test_data = pd.read_csv(test_file, sep="\t", header=0)

    train_X = train_data.drop("heart_disease", axis=1)
    train_y = train_data["heart_disease"]
    test_X = test_data.drop("heart_disease", axis=1)
    test_y = test_data["heart_disease"]

    # Evaluating LinearRegressor

    linreg = LinearRegressor(epochs=10, batch_size=5, learning_rate=0.01, seed=5)
    linreg.train(train_X, train_y)

    linreg_preds = linreg.predict(test_X)
    linreg_preds = linreg_preds > 0.5

    linreg_errors = sum(linreg_preds != test_y)
    linreg_error_rate = linreg_errors/len(test_y)
    print("linreg error:", linreg_error_rate)

    # Evaluating LogisticRegressor

    logreg = LogisticRegressor(thresh=0.5, epochs=10, batch_size=5, learning_rate=0.01, seed=4)
    logreg.train(train_X, train_y)

    logreg_preds = logreg.predict(test_X)

    logreg_errors = sum(logreg_preds != test_y)
    logreg_error_rate = logreg_errors/len(test_y)
    print("logreg error:", logreg_error_rate)

main()