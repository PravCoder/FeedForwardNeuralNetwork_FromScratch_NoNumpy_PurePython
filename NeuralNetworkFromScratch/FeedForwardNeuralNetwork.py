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

    def prepare_network(self):
        # initalize weights/biases
        for layer_indx in range(1, len(self.dimensions)):
            n = self.dimensions[layer_indx-1]
            m = self.dimensions[layer_indx]
            self.W[layer_indx] = np.random.randn(n, m)
            self.b[layer_indx] = np.random.rand(m)
            print(f"weights of layer-{layer_indx}: {self.W[layer_indx].shape}, bias: {self.b[layer_indx].shape}") # to check the shape of each layers weights

    def train(self):
        self.prepare_network()
        for iter_num in range(self.iterations):
            print(f"Iteration #{iter_num}")

if __name__ == "__main__":
    X = [[-4.0], [-3.0], [-2.0], [-1.5], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]
    Y = [[21.1], [4.3], [-1.8], [-1.5], [0.8], [1.2], [6.5], [19.3], [40.7], [68.5]]
    dims = [2, 3, 3, 1]
    acts = ["R","R","R","R","R"]
    net = Model(X, Y, dims, acts, iterations=1, learning_rate=0.01)
    net.train()
