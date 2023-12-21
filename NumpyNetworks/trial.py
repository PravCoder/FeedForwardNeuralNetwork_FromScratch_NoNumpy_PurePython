import math
import random

def sigmoid(z):
    return 1 / (1+math.exp(-z))
def d_sigmoid(z):
    return z * (1-z)
def init_weight():
    return random.uniform(-1, 1)


num_inputs = 2
num_hidden_nodes = 2
num_outputs = 1
num_training_sets = 4   # number of examples

def main():

    lr = 0.1

    hidden_layer = []   # length of num-hidden-nodes
    output_layer = []   # length of num-outputs in output-layer

    hidden_layer_bias = []  # length of num-hidden-nodes
    output_layer_bias = []  # length of num-output-nodes in output-layer

    hidden_weights = []  # 2D: [input-layer-node-i][hidden-layer-node-i], connection from [prev-node]-to-[cur-node], [num_inputs][num_hidden_nodes]
    output_weights = []  # 2D: [hidden-layer-node-i][output-layer-node-i], connection from [prev-node]-to-[cur-node], [num_hidden_nodes][num_output_nodes]

    training_inputs = [[0.0, 0.0],  # 2D: [input-example-i][input-node-value-i], index what example and then the node of that example in input-layer
                       [1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0]]
    
    training_outputs = [[0.0],  # 2D: [output-example-i][output-node-value-i], index what example and then the node of that example in output-layer
                       [1.0],
                       [1.0],
                       [0.0]]
    
    for i in range(num_inputs):     # iterate nodes in input-layer, no bias for input-layer
        hidden_weights.append([])
        for j in range(num_hidden_nodes):       # iterate nodes in hidden-layer
            hidden_weights[i].append(0)
            hidden_weights[i][j] = init_weight()        # init weight of hidden-weights[input-node-i][hidden-node-j]

    for i in range(num_hidden_nodes):       # iterate nodes in hidden-layer
        hidden_layer_bias.append(0)
        hidden_layer_bias[i] = init_weight()        # init bias for cur-node-i in hidden-layer
        output_weights.append([])
        for j in range(num_outputs):        # iterate nodes in output-layer
            output_weights[i].append(0)
            output_weights[i][j] = init_weight()    # init weight of output-weights[hidden-node-i][output-node-j]
        hidden_layer.append(0)

    for i in range(num_outputs):        # iterate nodes in output-layer
        output_layer_bias.append(0)
        output_layer.append(0)
        output_layer_bias[i] = init_weight()        # init bias for cur-node-i in output-layer 



    training_set_order = [0, 1, 2, 3]

    num_epochs = 10000      # number of iterations

    for epoch in range(num_epochs):     # iterate number of training epochs

        for x in range(num_training_sets):      # iterate 
            i = training_set_order[x]
            # FORWARD PASS

            # compute hidden-layer activation
            for j in range(num_hidden_nodes):       # iterate nodes in hidden-layer
                activation = hidden_layer_bias[j];      # init activation sum for cur-node-j in hidden-layer to bias of jth node in hidden-layer
                for k in range(num_inputs):     # iterate nodes in input-layer
                    activation += training_inputs[i][k] * hidden_weights[k][j]      # add to activation sum inputs[example-i][input-node-k] * hidden-weights[input-node-k][hidden-node-j]
                hidden_layer[j] = sigmoid(activation)       # set hidden-layer jth node activation to non-linearity of activation sum

            # compute output-layer activation
            for j in range(num_outputs):        # iterate nodes in output-layer
                activation = output_layer_bias[j]       # init activation sum cur-node-j in output-layer to bias of jth node in output-layer
                for k in range(num_hidden_nodes):       # iterate nodes in hidden-layer
                    activation += hidden_layer[k] * output_weights[k][j]    # add to activation sum # hidden-layer[node-k] * output-weights[hidden-node-k][output-node-j]
                output_layer[j] = sigmoid(activation)   # set output-layer jth node activation to non-linearity of activation sum


            print(f'Input: {training_inputs[i][0], training_inputs[i][1]}   Output: {output_layer[0]}   Expected: {training_outputs[i][0]}')
            # BACKPROPAGATION
                
            # compute change in each output-weight
            deltaOutput = []    # stores gradients of error with respect to nodes in output-layer, For each output node, it stores the product of the error and the derivative of the sigmoid activation function for that node.
            for j in range(num_outputs):        # iterate nodes in output-layer
                deltaOutput.append(0)
                errorOutput = training_outputs[i][j] - output_layer[j]      # compute difference in label-output[example-i][output-node-j] - output-layer[node-j]-activation
                deltaOutput[j] = errorOutput * d_sigmoid(output_layer[j])   # gradient of node-j in output-layer = error * sigmoid-derivative of output-layer[node-j
            
            # compute change in each hidden-weight
            deltaHidden = []    # stores gradients of error with respect to nodes in hidden layer
            for j in range(num_hidden_nodes):       # iterate nodes in hidden-layer
                deltaHidden.append(0)
                errorHidden = 0.0       #
                for k in range(num_outputs):
                    errorHidden += deltaOutput[k] * output_weights[j][k] 
                deltaHidden[j] = errorHidden * d_sigmoid(hidden_layer[j])

            # apply change in output-weights
            for j in range(num_outputs):
                output_layer_bias[j] = deltaOutput[j] * lr
                for k in range(num_hidden_nodes):
                    output_weights[k][j] += hidden_layer[k] * deltaOutput[j] * lr

            for j in range(num_hidden_nodes):
                hidden_layer_bias[j] += deltaHidden[j] * lr
                for k in range(num_inputs):
                    hidden_weights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr
main()

"""
XOR Function:
input  |   ouput
0 0    |   0
1 0    |   1
0 1    |   1
1 1    |   0
if the 2 inputs are equal outputs 0, if they differ output 1.

Input nodes: 2
Hidden Layers: 1 with 2 nodes
Output nodes: 1



"""