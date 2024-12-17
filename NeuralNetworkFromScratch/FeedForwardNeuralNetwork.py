import numpy as np

class Model:

    def __init__(self, X, Y, dimensions, activations, iterations=100, learning_rate=0.01):
        self.X = X                  # each row is an example, each element in row-example corresponds to a input node value
        self.Y = Y                  # each row is an output example label, each element in output-row-label corresponds to a output nodes label value
        self.dimensions = dimensions  # [1, 5,5,5 1] number of nodes per layer including input to output layers
        self.activations = activations  # ["R", "R","R","R", "S"] defines the activation used for each layer via the index, every layer as activation func
        self.iterations = iterations

        self.W = {}  # dict where each key is layer-index and its value is a matrix of weights for that index-layer
        self.b = {}  # dict where key is layer-indx and its value is a matrix of biases for that index-layer
        self.intermed = []  # list where each element is dict with intermediate values {layer_indx, A_prev, W, b, Z}

    # Activation Functions
    def ReLU_forward(self, Z):
        return np.maximum(0, Z)
    def ReLU_backward(self):
        pass
    def Sigmoid_forward(self, Z):
        Z = np.clip(Z, -500, 500)  
        return 1 / (1 + np.exp(-Z))
    # Loss Functions
    def binary_cross_entropy_forward(self, AL, Y):
        epsilon = 1e-15  
        AL = np.clip(AL, epsilon, 1 - epsilon)  # clip activations of last-layer min=epsilon max=1-epsilon, if values in AL are less than min it becomes min if values are more than max it becomes max
        # compute loss which is average of all examples Y-shape(examples, output-nodes), perform element-wise multiplication
        loss = -1 / Y.shape[0] * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return loss 

    def prepare_network(self):
        # initalize weights/biases
        print(f"dims: {self.dimensions}")
        for layer_indx in range(1, len(self.dimensions)):
            n = self.dimensions[layer_indx-1]
            m = self.dimensions[layer_indx]
            self.W[layer_indx] = np.random.randn(n, m)
            self.b[layer_indx] = np.random.rand(m)
            print(f"weights of layer-{layer_indx}: {self.W[layer_indx].shape}, bias: {self.b[layer_indx].shape}") # to check the shape of each layers weights

        # preset weights for testing purposes
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
        self.b = {1: np.array([0.38129563, 0.31356892, 0.16569252]), 2: np.array([0.53641851, 0.26092846, 0.36984623]), 3: np.array([0.37719807])}

    def forward_propagation(self):
        self.intermed = []
        # print(f"{self.W.keys()=}")
        # print(f"{self.b.keys()=}")
        A_prev = self.X

        # iterate from 1st layer not input because input layer doesnt have Z & A
        for layer_indx in range(1, len(self.dimensions)):
            print(f"-------Layer #{layer_indx}-------")
            # weighted-sum
            Z = np.dot(A_prev, self.W[layer_indx]) + self.b[layer_indx]
            print(f"{Z=}, {Z.shape}")
            # activation function
            if self.activations[layer_indx] == "R":
                A = self.ReLU_forward(Z)
                print(f"relu: {A=}, {A.shape}")
            if self.activations[layer_indx] == "S":
                A = self.Sigmoid_forward(Z)
                print(f"sig: {A=}, {A.shape}")
            # set next layers previous-activation to be cur-activation
            A_prev = A
            # print(f"{layer_indx=}")
            self.intermed.append({"layer":layer_indx,"A_prev": A_prev,"W": self.W[layer_indx], "b": self.b[layer_indx], "Z": Z})
        
        # return the alst value holded by A which is the activations of last-layer
        return A
        
    def train(self):
        self.prepare_network()
        # training main loop
        for iter_num in range(1, self.iterations):
            print(f"Iteration #{iter_num}")

            # propagate data forward through network
            AL = self.forward_propagation()
            print(f"AL: {AL.shape} {AL}")

            cost = self.binary_cross_entropy_forward(AL, self.Y)
            print(f"Cost: {cost}")

if __name__ == "__main__":
    # XOR function
    X = [[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2,0.3], [0.3,0.3], [0.4,0.5], [0.4,0.4], [0.5,0.6], [0.6,0.6], [0.6,0.7]]
    Y = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
    X1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y1 = [[0], [1], [1], [0]]
    dims = [2, 3, 3, 1]
    acts = ["R","R","R","S"]
    net = Model(np.array(X1), np.array(Y1), dims, acts, iterations=2, learning_rate=0.01)
    net.train()

"""
TODO:
- manually hand check first iterations Z values

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