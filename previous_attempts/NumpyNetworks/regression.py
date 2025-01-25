import numpy as np
import math
import random
import matplotlib.pyplot as plt 
from FNN import *

def generate_sine_data(num_samples=100, noise_factor=0.1):
    X = [random.uniform(0, 2 * math.pi) for _ in range(num_samples)]
    Y = [math.sin(x) + random.uniform(-noise_factor, noise_factor) for x in X]
    X = [[x] for x in X]
    Y = [[y] for y in Y]
    return X, Y

def reformat_data(x, y):
    train_x, train_y = [[]], [[]]
    for input in x:
        train_x[0].append(input)
    for output in y:
        train_y[0].append(output)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y
    
    
def main():
    layers_dims = [1, 32, 1]
    X, Y = generate_sine_data(num_samples=50, noise_factor=0.1)
    x_values = [x[0] for x in X]
    y_values = [y[0] for y in Y]
    train_x, train_y = reformat_data(x_values, y_values)
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.001, 500, regression=True)
    nn.train()

    predictions = []
    for input in x_values:
        print(nn.predict(np.array([[input]]), None)[0])
        predictions.append(nn.predict(np.array([[input]]), None)[0])

    # Plot the sine curve data
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', label='Noisy Sine Curve Data')
    plt.scatter(x_values, predictions, color='red', label='Network Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Sine Curve Data')
    plt.legend()
    plt.show()

main()


