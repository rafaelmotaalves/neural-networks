import numpy as np

def load_dataset():
    trainfile = np.load("data/train.npz")
    testfile = np.load("data/test.npz")
    valfile = np.load("data/val.npz")

    x_train, y_train = trainfile["x"], trainfile["y"]
    x_test, y_test = testfile["x"], testfile["y"]
    x_val, y_val = valfile["x"], valfile["y"]

    return x_train, x_val, x_test, y_train, y_val, y_test