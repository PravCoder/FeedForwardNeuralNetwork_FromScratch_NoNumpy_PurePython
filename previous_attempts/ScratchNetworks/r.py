import math
import matplotlib.pyplot as plt 
from NN import *
import math
import random

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
    return train_x, train_y
    

def main():
    layers_dims = [1, 10,10, 1]
    X, Y = generate_sine_data(num_samples=1000, noise_factor=0.1)
    print(X)
    # Extract x values from X and y values from Y for plotting
    x_values = [x[0] for x in X]
    y_values = [y[0] for y in Y]
    train_x, train_y = reformat_data(x_values, y_values)

    nn = FeedForwardNeuralNetwork(train_x, train_y,   layers_dims, 0.01, 100, regression=True)
    nn.train()

    predictions = []
    for input in x_values:
        predictions.append(nn.predict([[input]])[0])

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


