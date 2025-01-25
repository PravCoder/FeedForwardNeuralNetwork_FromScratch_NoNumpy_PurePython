from FNN import *
import math
import random


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
    layers_dims = [1,32,1]
    # get raw dataset in 1D-lists
    rawX, rawY = generate_sine_data(num_samples=50)
    # format data-x by hvaing eahc example in its own list and add that list to hte big list, format data-y by again adding the y-values to 1D-list
    train_x, train_y = reformat_data(rawX, rawY)

    n = FeedForwardNeuralNetwork(train_x, 
                                  train_y, 
                                  layers_dims, 
                                  0.001, 400, 
                                  binary_classification=False,
                                  multiclass_classification=False,
                                  regression=True,
                                  optimizer="gradient descent", 
                                  learning_rate_decay=False, 
                                  gradient_descent_variant="batch")

    n.train()


    graph_x_labels = [x for x in train_x[0]]
    graph_y_labels = [y for y in train_y[0]]
    network_predictions = []
    for i, x in enumerate(graph_x_labels):
        network_predictions.append(n.predict([[x]], None))

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