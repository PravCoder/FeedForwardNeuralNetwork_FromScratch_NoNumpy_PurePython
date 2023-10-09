import numpy as np
import math
import matplotlib.pyplot as plt 
from FNN import *


def func(x):
    return math.sin(x) # for different function
def proccess_data(points):
    train_x = [[]]
    train_y = [[]]
    x = 0
    while x < points:
        input_val = x/100
        train_x[0].append(input_val)
        train_y[0].append(func(input_val))
        x += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y
    

def main():
    layers_dims = [1, 1]
    x, y = proccess_data(5000)
    train_x = x
    train_y = y
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.001, 500, regression=True)
    nn.train()

    inputs = [] # stores each input to function in 1D-list
    network_preedictions = []       # stores each prediction of network in 1D-list
    labels = []     # stores each actual label of function in 1D-list
    for e in range(len(train_x[0])):    # iterate through example indicies
        cur_input = train_x[0][e]   # get current example input value
        inputs.append(cur_input)        
        labels.append(train_y[0][e])    # get current example label using example index
        preds = nn.predict([[cur_input]])
        network_preedictions.append(preds[0])     # add prediction of network to nerual network 
    
    #print("x: " + str(train_x))
    #print("network-predictions: " + str(network_preedictions))
    #print("labels: " + str(labels))
    #print(len(inputs) == len(labels))
    #print(len(inputs) == len(network_preedictions))
    plt.plot(inputs, labels, color="red")   # Sine curve
    plt.plot(inputs, network_preedictions, color="blue")    # network fit
    plt.xlim(-1.8, -0.7)
    plt.ylim(0, 200)
    plt.show()

main()


