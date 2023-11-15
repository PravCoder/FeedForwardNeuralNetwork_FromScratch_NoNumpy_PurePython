# from NN import *
from NumpyNetworks.FFN_Coursera1 import *
import numpy as np
import math
import random

# NOTE: The purpose of this file is for debugging the implementation. It runs the model on simple function
# like y=0/5x. It also has a implementation of a linear regression model to also compare
# the neural network model with.


# def func(x):
#     return math.sin(x)     # for different function
# def proccess_data():
#     train_x = [[]]
#     train_y = [[]]
#     x = 0
#     while x < 1000:
#         input_val = x/100
#         train_x[0].append(input_val)
#         train_y[0].append(func(input_val))
#         x += 0.5

#     # Convert train_x and train_y to NumPy arrays
#     train_x = np.array(train_x)
#     train_y = np.array(train_y)

#     # Normalize the data
#     mean = np.mean(train_x)
#     std = np.std(train_x)
#     train_x = (train_x - mean) / std

#     return train_x, train_y

# def plot_fit(nn, train_x, train_y): 
#     inputs = [] # stores each input to function in 1D-list
#     network_preedictions = []       # stores each prediction of network in 1D-list
#     labels = []     # stores each actual label of function in 1D-list
#     for e in range(len(train_x[0])):    # iterate through example indicies
#         cur_input = train_x[0][e]   # get current example input value
#         inputs.append(cur_input)        
#         labels.append(train_y[0][e])    # get current example label using example index
#         network_preedictions.append(nn.predict([[cur_input]], [], False)[0])     # add prediction of network to nerual network 
    
#     #print("x: " + str(train_x))
#     #print("network-predictions: " + str(network_preedictions))
#     #print("labels: " + str(labels))
#     #print(len(inputs) == len(labels))
#     #print(len(inputs) == len(network_preedictions))
#     plt.plot(inputs, labels, color="red")   # Sine curve
#     plt.plot(inputs, network_preedictions, color="blue")    # network fit
#     plt.xlim(-0.1, 1.2)
#     plt.ylim(-1.0, 1.0)
#     plt.show()

# def main():
#     layers_dims = [1,15,5,5,1]
#     train_x, train_y = proccess_data()
#     nn = FeedForwardNeuralNetwork(train_x, 
#                                   train_y, 
#                                   layers_dims, 
#                                   0.0075, 5, 
#                                   l2_regularization=True,
#                                   binary_classification=False,
#                                   multiclass_classification=False,
#                                   regression=True,
#                                   optimizer="gradient descent", 
#                                   learning_rate_decay=False, 
#                                   gradient_descent_variant="batch")
#     # Run methods
#     nn.train()
#     #nn.evaluate_accuracy()
#     plot_fit(nn, train_x, train_y)
# # main()



def generate_sine_data(num_samples=100, noise_factor=0.1):
    X = [random.uniform(0, 2 * math.pi) for _ in range(num_samples)]
    Y = [math.sin(x) + random.uniform(-noise_factor, noise_factor) for x in X]
    X = [x for x in X]
    Y = [y for y in Y]
    return X, Y

def reformat_data(x, y):
    train_x, train_y = [[]], [[]]
    for i, input in enumerate(x):
        train_x[0].append(input)
    for j, output in enumerate(y):
        train_y[0].append(output)

    return train_x, train_y

def main_fit_curve():
    layers_dims = [1,32,32,32, 1]
    # get raw dataset in 1D-lists
    rawX, rawY = generate_sine_data(num_samples=50)
    # format data-x by hvaing eahc example in its own list and add that list to hte big list, format data-y by again adding the y-values to 1D-list
    train_x, train_y = reformat_data(rawX, rawY)

    n = FeedForwardNN(np.array(train_x), np.array(train_y), layers_dims, 0.001, 1000)

    n.model()


    graph_x_labels = [x for x in train_x[0]]
    graph_y_labels = [y for y in train_y[0]]
    network_predictions = []
    for i, x in enumerate(graph_x_labels):
        network_predictions.append(n.predict(np.array([[x]])))

    # print("predictions: "+str(network_predictions))
    # print(len(graph_x_labels) == len(network_predictions))
    # print(len(graph_x_labels) == len(graph_y_labels))
    # print("X: "+str(train_x))
    # print("x-vals: "+str(graph_x_labels))
    # Plot the sine curve data
    plt.figure(figsize=(8, 6))
    plt.scatter(graph_x_labels, graph_y_labels, color='blue', label='Noisy Sine Curve Data')
    plt.scatter(graph_x_labels, network_predictions, color='red', label='Network Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Sine Curve Data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
   main_fit_curve()