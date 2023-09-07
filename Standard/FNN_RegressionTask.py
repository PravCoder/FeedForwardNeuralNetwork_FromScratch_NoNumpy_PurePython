from FNN import *
import numpy as np
import math

# NOTE: The purpose of this file is for debugging the implementation. It runs the model on simple function
# like y=0/5x. It also has a implementation of a linear regression model to also compare
# the neural network model with.


def func(x):
    return math.sin(x)      # for different function
def proccess_data():
    train_x = [[]]
    train_y = [[]]
    for x in range(0, 101):
        train_x[0].append(x/100)
        train_y[0].append(func(x))
    return train_x, train_y
def plot_fit(nn, train_x, train_y): 
    inputs = [] # stores each input to function in 1D-list
    network_preedictions = []       # stores each prediction of network in 1D-list
    labels = []     # stores each actual label of function in 1D-list
    for e in range(len(train_x[0])):    # iterate through example indicies
        cur_input = train_x[0][e]   # get current example input value
        inputs.append(cur_input)        
        labels.append(train_y[0][e])    # get current example label using example index
        network_preedictions.append(nn.predict([[cur_input]], [], True)[0])     # add prediction of network to nerual network 
    
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

def main():
    layers_dims = [1,15,5,5,1]
    train_x, train_y = proccess_data()
    nn = FeedForwardNeuralNetwork(train_x, 
                                  train_y, 
                                  layers_dims, 
                                  0.0075, 100, 
                                  l2_regularization=False,
                                  binary_classification=False,
                                  multiclass_classification=False,
                                  regression=True,
                                  optimizer="gradient descent", 
                                  learning_rate_decay=False, 
                                  gradient_descent_variant="batch")
    # Run methods
    nn.train()
    preds = nn.predict([[1]], [], True)
    print(preds)

    plot_fit(nn, train_x, train_y)
main()



"""
#-----------------------------------
# LINEAR REGRESSION:
# function y = 10x
X = np.array([1/100,2/100,3/100,4/100])
Y = np.array([10/100,20/100,30/100,40/100])

# prediction using nn
def forward(w, x):
    y = w*x
    return y
# loss
def compute_mse(y, y_pred):
    mse = np.mean(y-y_pred)**2
    return mse
# gradient
# f = (y-y_pred) ** 2
# f(w, x) = (y - wx) ** 2
# df/dw = 2 * (y - wx) * x
# df/dw = 2 * (y - y_pred) * x
def gradient(x, y, y_pred):
    return np.mean(2 * (y - y_pred) * x)
w = 0
def main_linear_reg(w):
    learning_rate = 0.1
    steps = 1

    for step in range(1, steps+1):
        y_pred = forward(w,X)

        loss = compute_mse(Y, y_pred)

        grad = gradient(X, Y, y_pred)
        
        w += grad * learning_rate

        if True:
            print(f"y_pred {y_pred}")
            print(f"grad = {grad}")
            print(f"step {step}: weight = {w:3f} loss= {loss:3f}")
            print("-----------")

# Run Linear Regression Model
print("------------\n"+"LINEAR REGRESSION MODEL:")
main_linear_reg(w)
trained_guess = forward(w, 40)
#print(f"f(40) using nn: {trained_guess:3f}")

# Run NN
print("------------\n"+"NEURAL NETWORK MODEL:")
main()"""




