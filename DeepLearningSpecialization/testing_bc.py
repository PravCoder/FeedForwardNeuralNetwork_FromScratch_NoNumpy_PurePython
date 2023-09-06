from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from FNN import FeedForwardNeuralNetwork

def process_data():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    examples = len(X_train) # = len(Y_train), Y_train = [1,0,1,0,..,m]
    x_nodes = len(X_train[0])
    y_nodes = 1
    proccessed_X_train = []
    proccessed_Y_train = []
    for n in range(x_nodes):
        proccessed_X_train.append([])
        for e in range(examples):
            proccessed_X_train[n].append(X_train[e][n])
    for n in range(y_nodes):
        proccessed_Y_train.append([])
        for e in range(examples):
            proccessed_Y_train[n].append(Y_train[e])

    return np.array(proccessed_X_train), np.array(proccessed_Y_train)

if __name__ == "__main__":

    layers_dims = [30, 2, 4, 3, 1] 
    train_x, train_y = process_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500)
    nn.train()


