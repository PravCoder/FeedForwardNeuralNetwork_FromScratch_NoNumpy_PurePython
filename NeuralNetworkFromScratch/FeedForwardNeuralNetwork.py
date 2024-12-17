import numpy as np

class Model:

    def __init__(self, X, Y, dimensions, activations, iterations=100, learning_rate=0.01):
        self.X = X                  # each row is an example, each element in row-example corresponds to a input node value
        self.Y = Y                  # each row is an output example label, each element in output-row-label corresponds to a output nodes label value
        self.dimensions = dimensions  # [1, 5,5,5 1] number of nodes per layer including input to output layers
        self.activations = activations  # ["R", "R","R","R", "S"] defines the activation used for each layer via the index, every layer as activation func
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.W = {}  # dict where each key is layer-index and its value is a matrix of weights for that index-layer
        self.b = {}  # dict where key is layer-indx and its value is a matrix of biases for that index-layer
        self.intermed = []  # list where each element is dict with intermediate values {layer_indx, A_prev, W, b, Z}
        self.dW = {}
        self.db = {}

    # Activation Functions
    def ReLU_forward(self, Z):
        return np.maximum(0, Z)
    def ReLU_backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ
    def Sigmoid_forward(self, Z):
        Z = np.clip(Z, -500, 500)  
        return 1 / (1 + np.exp(-Z))
    def Sigmoid_backward(self, Z, dA=1):
        s = self.Sigmoid_forward(Z)
        dZ = s * (1 - s)
        return dA * dZ
    # Loss Functions
    def binary_cross_entropy_forward(self, AL, Y):
        epsilon = 1e-15  
        AL = np.clip(AL, epsilon, 1 - epsilon)  # clip activations of last-layer min=epsilon max=1-epsilon, if values in AL are less than min it becomes min if values are more than max it becomes max
        # compute loss which is average of all examples Y-shape(examples, output-nodes), perform element-wise multiplication
        loss = -1 / Y.shape[0] * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return loss 
    def binary_cross_entropy_backward(self, AL, Y):
            epsilon = 1e-15  
            AL = np.clip(AL, epsilon, 1 - epsilon)  
            dAL = - (Y / AL) + (1 - Y) / (1 - AL)   # derivative of 
            return dAL / Y.shape[0]

    def prepare_network(self):
        # initalize weights/biases
        print(f"dims: {self.dimensions}")
        for layer_indx in range(1, len(self.dimensions)):
            n = self.dimensions[layer_indx-1] # number of nodes in previous-layer
            m = self.dimensions[layer_indx]   # number of nodes in current-layer
            self.W[layer_indx] = np.random.randn(n, m) # each row is connection from prev-layer-node, each col in that row is connection from prev-layer-node to cur-layer-node
            self.b[layer_indx] = np.random.rand(m)
            print(f"weights of layer-{layer_indx}: {self.W[layer_indx].shape}, bias: {self.b[layer_indx].shape}") # to check the shape of each layers weights

        # preset weights for testing purposes
        # self.W = {1: np.array(
        #     [[-0.15711978,  1.91467883,  1.30879812],
        #     [-0.40953811, -1.25633144,  1.39848743]]), 
                
        #     2: np.array(
        #     [[ 1.30957982, -0.58575335,  0.27496437],
        #     [ 0.0615504 ,  0.38399249,  0.67944275],
        #     [ 1.27965209,  0.87114635, -0.02316647]]), 

        #     3: np.array(
        #     [[-0.69113884],
        #     [ 0.18692256],
        #     [-1.07356081]])
        #     }
        # self.b = {1: np.array([0.38129563, 0.31356892, 0.16569252]), 2: np.array([0.53641851, 0.26092846, 0.36984623]), 3: np.array([0.37719807])}

    def forward_propagation(self):
        self.intermed = []
        # print(f"{self.W.keys()=}")
        # print(f"{self.b.keys()=}")
        A_prev = self.X

        # iterate from 1st layer not input because input layer doesnt have Z & A
        for layer_indx in range(1, len(self.dimensions)):
            #print(f"-------Layer #{layer_indx}-------")
            # weighted-sum
            Z = np.dot(A_prev, self.W[layer_indx]) + self.b[layer_indx]
            #print(f"{Z=}, {Z.shape}")
            # activation function
            if self.activations[layer_indx] == "R":
                A = self.ReLU_forward(Z)
                #print(f"relu: {A=}, {A.shape}")
            if self.activations[layer_indx] == "S":
                A = self.Sigmoid_forward(Z)
                #print(f"sig: {A=}, {A.shape}")
            # set next layers previous-activation to be cur-activation
            A_prev = A
            # print(f"{layer_indx=}")
            self.intermed.append({"layer":layer_indx,"A_prev": A_prev,"W": self.W[layer_indx], "b": self.b[layer_indx], "Z": Z})
        
        # return the last value holded by A which is the activations of last-layer
        return A
    
        
    def train(self):
        self.prepare_network()
        # training main loop
        for iter_num in range(1, self.iterations):
            # propagate data forward through network
            AL = self.forward_propagation()
            #print(f"AL: {AL.shape} {AL}")

            cost = self.binary_cross_entropy_forward(AL, self.Y)

            self.dW, self.db = self.backward_propagation(AL, self.Y)

            self.W, self.b = self.optimizer_update()
            print(f"Cost #{iter_num} is {cost}")


    def backward_propagation(self, AL, Y):
        self.dW, self.db = self.reset_grads()
        # derivative of cost-func respect to final activation, make Y shape shape as AL/
        dA_prev = self.binary_cross_entropy_backward(AL, Y.reshape(AL.shape)) 

        # iterate from last-layer-indx to 1st-layer, excluding input-layer-0 doesnt need backprop
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            # get the cache we stored for cur-layer in forward-prop
            cache = self.get_cache(layer_indx)
            # unpack cache and get A_prev, W, b, Z used for the cur-layer
            A_prev, W, b, Z = cache["A_prev"], cache["W"], cache["b"], cache["Z"]
            # compute gradients of cur-layer, pass in prev-layer-dA and set it for next-layer
            dA_prev, dW, db = self.layer_backward(dA_prev, A_prev, W, b, Z, layer_indx)
            # set save gradients of cur-layer
            self.dW[layer_indx] = dW
            self.db[layer_indx] = db
        
        return self.dW, self.db

    def layer_backward(self, dA, A_prev, W, b, Z, layer_indx):
        m = A_prev.shape[0]
        if self.activations[layer_indx] == "R":
            dZ = self.ReLU_backward(Z, dA) 
        if self.activations[layer_indx] == "S":
            dZ = self.Sigmoid_backward(Z, dA)
        
        print(f"Layer {layer_indx}: dZ.shape = {dZ.shape}, A_prev.shape = {A_prev.shape}")

        dW = 1 / m * np.dot(A_prev.T, dZ)               # compute derivative of loss respect to weights of cur-layer
        print(f"{layer_indx=} - {dW.shape=} - {dZ.shape=}")
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True)  # compute derivative of loss respect to bias of cur-layer
        dA_prev = np.dot(dZ, W.T)                       # compute derivative of loss respect to activation of previous-layer
        return dA_prev, dW, db
    
    def optimizer_update(self):
        # iterate from 1st-layer to last-layer because input-layer doesnt have weights to update
        for key, val in self.dW.items():
                print(f"dW-{key=}, {val.shape=}")

        for layer_indx in range(1, len(self.dimensions)):
            print(f"Layer {layer_indx}: W.shape = {self.W[layer_indx].shape}, dW.shape = {self.dW[layer_indx].shape}")
            self.W[layer_indx] -= self.learning_rate * self.dW[layer_indx]
            self.b[layer_indx] -= self.learning_rate * self.db[layer_indx]
        return self.W, self.b  # returning for precaution
    
    def reset_grads(self):
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            n = self.dimensions[layer_indx-1] # number of nodes in previous-layer
            m = self.dimensions[layer_indx]   # number of nodes in current-layer
            self.dW[layer_indx]  = np.zeros((n, m))
            self.db[layer_indx] = np.zeros(m)
        return self.dW, self.db

    def get_cache(self, layer_indx):
        for cache in self.intermed:
            if cache["layer"] == layer_indx:
                return cache

if __name__ == "__main__":
    # XOR function
    X = [[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2,0.3], [0.3,0.3], [0.4,0.5], [0.4,0.4], [0.5,0.6], [0.6,0.6], [0.6,0.7]]
    Y = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
    X1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y1 = [[0], [1], [1], [0]]
    dims = [2, 3, 3, 1]
    acts = ["R","R","R","S"]
    net = Model(np.array(X1), np.array(Y1), dims, acts, iterations=5, learning_rate=0.01)
    net.train()

"""
TODO:
- manually hand check backprop matrix dimensions
STATUS:
here is my attempt at a neural network library from scratch using numpy, but i keep getting this error where my graidents dW and weights W do not have the smae shape i think my dW shape is getting messed somewhere i could be wrong, also my weights for a single layer are (previous nodes, current nodes. The layer index start at zero fro the input layer. What is wrong with that it is giving me this output and error.

# preset weights for testing purposes, and check in test.py
Weights
self.W = {1: np.array(
            [[-0.15711978,  1.91467883,  1.30879812],
            [-0.40953811, -1.25633144,  1.39848743]]), 
                
            2: np.array(
            [[ 1.30957982, -0.58575335,  0.27496437],
            [ 0.0615504 ,  0.38399249,  0.67944275],
            [ 1.27965209,  0.87114635, -0.02316647]]), 

            3: np.array(
            [[-0.69113884],
            [ 0.18692256],
            [-1.07356081]])
            }

Biases
self.b = {1: np.array([0.38129563, 0.31356892, 0.16569252]), 2: np.array([0.53641851, 0.26092846, 0.36984623]), 3: np.array([0.37719807])}
"""