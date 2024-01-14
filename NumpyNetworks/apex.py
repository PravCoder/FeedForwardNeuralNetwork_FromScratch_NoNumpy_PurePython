import math
import random
import numpy as np


class Net:

    def __init__(self, train_x, train_y, dimensions, lr, epochs):
        self.train_x = train_x
        self.train_y = train_y
        self.dimensions = dimensions
        self.lr = lr
        self.epochs = epochs

        self.activations = {} # {layer-indx: [n1, n2, n3]}, input layer-indx-0 not included
        self.weights = {}     # {layer-indx: 2D[[],[]]: [prev-layer-node-i][cur-layer-node-i]}, input-layer doesn't have weights
        self.bias = {}        # {layer-indx: 2D[[]]}, [layer-indx][node]
        self.grads = {}     # {layer-indx: [deltaN1, deltaN2, deltaN3]}, only store gradients for output/hidden layers

    def init_weight(self):
        return random.uniform(-1, 1)
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    def d_sigmoid(self, z):
        return z * (1-z)
    def relu(self, z):
        return max(0, z)
    def d_relu(self, z):
        return 1 if z > 0 else 0
    
    def init_activations(self):
        for layer_indx in range(1, len(self.dimensions)):
            self.activations[layer_indx] = []
            for node in range(self.dimensions[layer_indx]):
                self.activations[layer_indx].append(0)

        # print(f'activations: {self.activations}')

    def init_weights_bias(self):
        for layer_indx in range(1, len(self.dimensions)):
            self.weights[layer_indx] = []
            self.bias[layer_indx] = [self.init_weight() for _ in range(self.dimensions[layer_indx])]
            for prev_node_i in range(self.dimensions[layer_indx-1]):
                self.weights[layer_indx].append([])
                for cur_node_i in range(self.dimensions[layer_indx]):
                    self.weights[layer_indx][prev_node_i].append(self.init_weight())
        # print(f'weights: {self.weights}')
        # print(f'bias: {self.bias}')

    def init_gradients_weights_bias(self):
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            self.grads[layer_indx] = []
            for node in range(self.dimensions[layer_indx]):
                self.grads[layer_indx].append(0)
        # print(f'grads: {self.grads}')

    def forward_pass(self, example):
        # compute hidden layers activations
        for layer_indx in range(1, len(self.dimensions)):

            if layer_indx == 1:
                for cur_node in range(self.dimensions[layer_indx]):
                    activation = self.bias[layer_indx][cur_node]
                    for input_node in range(self.dimensions[layer_indx-1]):
                        activation += self.train_x[example][input_node] * self.weights[layer_indx][input_node][cur_node]
                    self.activations[layer_indx][cur_node] = self.sigmoid(activation)

            if layer_indx != 1:
                for cur_node in range(self.dimensions[layer_indx]):
                    activation = self.bias[layer_indx][cur_node]
                    for prev_node in range(self.dimensions[layer_indx-1]):
                        activation += self.activations[layer_indx-1][prev_node] * self.weights[layer_indx][prev_node][cur_node]
                    # TBD: Check for last layer different activation
                    self.activations[layer_indx][cur_node] = self.sigmoid(activation)
        # print("\nforward pass")
        # print(f'activations: {self.activations}')

    def backward_pass(self, example):
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            if layer_indx == len(self.dimensions)-1:
                for output_node in range(self.dimensions[layer_indx]):
                    errorOutput = self.train_y[example][output_node] - self.activations[layer_indx][output_node]
                    self.grads[layer_indx][output_node] = errorOutput * self.d_sigmoid(self.activations[layer_indx][output_node])
            
            if layer_indx != len(self.dimensions)-1:
                for cur_node in range(self.dimensions[layer_indx]):
                    errorHidden = 0.0
                    for after_node in range(self.dimensions[layer_indx+1]):
                        errorHidden += self.grads[layer_indx+1][after_node] * self.weights[layer_indx+1][cur_node][after_node]
                    self.grads[layer_indx][cur_node] = errorHidden * self.d_sigmoid(self.activations[layer_indx][cur_node])
        
        # print("\nbackward pass")
        # print(f'grads: {self.grads}')

    def update(self, example):
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            if layer_indx != 1:
                for cur_node in range(self.dimensions[layer_indx]):
                    self.bias[layer_indx][cur_node] += self.grads[layer_indx][cur_node] * self.lr
                    for prev_node in range(self.dimensions[layer_indx-1]):
                        self.weights[layer_indx][prev_node][cur_node] += self.activations[layer_indx-1][prev_node] * self.grads[layer_indx][cur_node] * self.lr
            if layer_indx == 1:
                for cur_node in range(self.dimensions[layer_indx]):
                    self.bias[layer_indx][cur_node] += self.grads[layer_indx][cur_node] * self.lr
                    for input_node in range(self.dimensions[layer_indx-1]):
                        # print(f'b: {self.weights[layer_indx][input_node][cur_node]}')
                        self.weights[layer_indx][input_node][cur_node] += self.train_x[example][input_node] * self.grads[layer_indx][cur_node] * self.lr
                        # print(f'a: {self.weights[layer_indx][input_node][cur_node]}, {self.train_x[example][input_node] * self.grads[layer_indx][cur_node] * self.lr}\n') # tha last {} is zero so weights might be updating because input is 0
        # print("\nupdate")
        # print(f'weights: {self.weights}')
        # print(f'bias: {self.bias}')

    def predict(self, example):
        last_layer = len(self.dimensions)-1
        # print(f'Input: {self.train_x[example]}   Output: {self.activations[last_layer][0]}   Expected: {self.train_y[example]}')
        print(f'Output: {self.activations[last_layer][0]}   Expected: {self.train_y[example]}')

    def model(self):
        """
        print("initalization")
        self.init_activations()
        self.init_weights_bias()
        self.init_gradients_weights_bias()
        self.forward_pass(0)    # SGD: pass each example at a time and update parameters
        self.backward_pass(0)
        self.update(0)"""

        print("initalization")
        self.init_activations()
        self.init_weights_bias()
        self.init_gradients_weights_bias()
        for epoch in range(self.epochs):
            for example in range(len(self.train_x)):
                self.forward_pass(example)    # SGD: pass each example one at a time and update parameters
                
                
                
                self.backward_pass(example)
                self.update(example)
        # after training forward pass and predict each example
        for example in range(len(self.train_x)):
            self.forward_pass(example)
            self.predict(example)


if __name__ == "__main__":
    # CREATE MODEL
    train_x = [[0.0, 0.0], 
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]]
    train_y = [[0.0], 
            [1.0],
            [1.0],
            [0.0]]
    dimensions = [2, 10,10, 1]
    nn = Net(train_x, train_y, dimensions, 0.1, 10000)
    nn.model()


"""
[DONE]: Dynamic layers
[DONE]: Binary classification
[TBD]: Noramal GD Instead of stochastic GD, so pass all of the examples at once then update
[TBD]: Different cost functions how that will affect backprop
[TBD]: Different activations and their derivatives in backprop
[TBD]: Check for last layer different activation

https://sebastianraschka.com/faq/docs/gradient-optimization.html


print("initalization")
        self.init_activations()
        self.init_weights_bias()
        self.init_gradients_weights_bias()
        for epoch in range(self.epochs):
            for example in range(len(self.train_x)):
                self.forward_pass(example)    # SGD: pass each example at a time and update parameters
                self.backward_pass(example)
                self.update(example)
"""