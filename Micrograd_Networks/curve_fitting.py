import random
import math
import matplotlib.pyplot as plt
from micrograd_network import MLP

def generate_sine_data(num_samples=100, noise_factor=0.1):
    X = [random.uniform(0, 2 * math.pi) for _ in range(num_samples)]
    Y = [math.sin(x) + random.uniform(-noise_factor, noise_factor) for x in X]
    X = [x for x in X]
    Y = [y for y in Y]
    return X, Y

def reformat_data(x, y):
    train_x, train_y = [], []
    for i, input in enumerate(x):
        train_x.append([])
        train_x[i].append(input)
    for j, output in enumerate(y):
        train_y.append([])
        train_y[j].append(output)

    return train_x, train_y

def main_fit_curve():
    # get raw dataset in 1D-lists
    rawX, rawY = generate_sine_data(num_samples=50)
    # format data-x by hvaing eahc example in its own list and add that list to hte big list, format data-y by again adding the y-values to 1D-list
    train_x, train_y = reformat_data(rawX, rawY)

    n = MLP(len(train_x[0]), [32, 1], -0.001, train_x, train_y, num_iterations=2500) # 0.001

    n.train()

    print("X: "+str(len(train_x)))
    print("Y: "+str(len(train_y)))
    graph_x_labels = [x[0] for x in train_x]
    graph_y_labels = [y for y in train_y]
    network_predictions = [value_obj[0].data for value_obj in n.predict(train_x)]

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