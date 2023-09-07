from FFN_Coursera2 import FeedForwardNeuralNetwork
import numpy as np
import math
import matplotlib.pyplot as plt 

# NOTE: The purpose of this file is for debugging the implementation. It runs the model on simple function
# like y=0/5x. It also has a implementation of a linear regression model to also compare
# the neural network model with.


def func(x):
    return math.sin(x)     # for different function
def proccess_data():
    train_x = [[]]
    train_y = [[]]
    x = 0
    while x < 1000:
        input_val = x/100
        train_x[0].append(input_val)
        train_y[0].append(func(input_val))
        x += 0.5
    print(train_x)
    return train_x, train_y
    

def main():
    layers_dims = [1,15,15,15,15,15,15,5,5,1]
    x, y = proccess_data()
    train_x = np.array(x)
    train_y = np.array(y)
    nn = FeedForwardNeuralNetwork(train_x, 
                                  train_y, 
                                  layers_dims, 
                                  0.0075, 2500, 
                                  regression=True
                                )
    # Run methods
    nn.train()
    nn.accuracy(train_x, train_y)

    inputs = [] # stores each input to function in 1D-list
    network_preedictions = []       # stores each prediction of network in 1D-list
    labels = []     # stores each actual label of function in 1D-list
    for e in range(len(train_x[0])):    # iterate through example indicies
        cur_input = train_x[0][e]   # get current example input value
        inputs.append(cur_input)        
        labels.append(train_y[0][e])    # get current example label using example index
        preds = nn.predict([[cur_input]], [])
        network_preedictions.append(preds[0])     # add prediction of network to nerual network 
    
    #print("x: " + str(train_x))
    #print("network-predictions: " + str(network_preedictions))
    #print("labels: " + str(labels))
    #print(len(inputs) == len(labels))
    #print(len(inputs) == len(network_preedictions))
    plt.plot(inputs, labels, color="red")   # Sine curve
    plt.plot(inputs, network_preedictions, color="blue")    # network fit
    plt.xlim(-0.1, 1.2)
    plt.ylim(-1.0, 1.0)
    plt.show()

main()


