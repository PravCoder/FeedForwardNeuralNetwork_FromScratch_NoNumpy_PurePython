import math as math
import matplotlib.pyplot as plt 
import numpy as np


class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, dimensions, learning_rate=0.0075, num_iterations=2500):
        self.X = X
        self.Y = Y
        self.m = len(self.X[0])
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost = 0
        self.weights = []
        self.inputs = []
        self.Z = []

    def initialize_weights(self):
        for l in range(1, len(self.dimensions)):
            for cur in range(self.dimensions[l]):
                for prev in range(self.dimensions[l-1]):
                    self.weights.append(W(l, prev, cur))

    def initialize_inputs(self):
        for r in range(len(self.X)):
            for e in range(len(self.X[r])):
                self.inputs.append(X(r, e, self.X[r][e]))

    def initialize_sums(self): 
        for l in range(1, len(self.dimensions)):
            for node in range(self.dimensions[l]):
                for example in range(0, self.m):
                    print(example)
                    self.Z.append(Z(l, node, example))

    def forward_propagation(self):
        for l in range(len(self.dimensions)):
            for n in range(self.dimensions[l]):
                for e in range(self.m):
                    self.get_Z(l,n,e).val = 0

    def get_Z(self, layer, node, example):
        for z in self.Z:
            if z.layer == layer and z.node == node and z.example == example:
                return z
    def get_X(self):
        pass
            




class W:
    def __init__(self, layer, from_node, to_node, val=0.01):
        self.layer = layer
        self.from_node = from_node
        self.to_node = to_node
        self.val = val
class Z:
    def __init__(self, layer, node, example):
        self.layer = layer
        self.node = node
        self.example = example
        self.val  = 0
class A:
    def __init__(self, layer, node, example):
        self.layer = layer
        self.node = node
        self.example = example
        self.val = 0

class X:
    def __init__(self, feature, example, val):
        self.feature = feature
        self.example = example
        self.val = val








layers_dims = [2, 1, 1] 
train_x = [
    [0.1, 0.2],
    [0.09, 0.08],
  
]
train_y = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
]

if __name__ == "__main__":
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500)
    nn.initialize_inputs()
    nn.initialize_sums()
    nn.initialize_weights()
    print("M: " + str(nn.m))
    print("Inputs: " + str([w.val for w in nn.inputs]))
    print("Weights: " + str([w.val for w in nn.weights]))
    print("Z's: " + str([z.val for z in nn.Z]))

    """iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, 2500)
    plt.ylim(0.68, 0.70)
    plt.show()"""


