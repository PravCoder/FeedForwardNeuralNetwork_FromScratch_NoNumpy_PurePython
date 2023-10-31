import random
import math
from Micrograd_Networks.micrograd_network import MLP


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
    for output in y:
        train_y.append(output)

    return train_x, train_y


def main():
    raw_x, raw_y = generate_sine_data()
    print(raw_x)


main()