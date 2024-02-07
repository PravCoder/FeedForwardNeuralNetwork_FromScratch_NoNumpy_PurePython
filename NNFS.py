import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(prev_layer_size, layer_size, initializer):
    W_layer = initializer(prev_layer_size, layer_size)    # call initalizer function
    b_layer = np.zeros((1, layer_size))     # a single column of zeros for biases because each node in cur-layer has bias
    W_layer, b_layer
    return W_layer, b_layer

class Layer:
    """
    Representation of a layer. Stores number of nodes, activation object, initializer function.
    """
    def __init__(self, num_nodes, activation, initializer):
        self.num_nodes = num_nodes
        self.activation = activation
        self.initializer = initializer

    def forward(self, A_prev, W, b):    # given activations for previous layer and weights and bias of current layer
        # print(A_prev)
        # print("\n")
        # print(W)
        Z = np.dot(A_prev, W) + b       # compute weighted-sum
        A = self.activation.forward(Z)  # activation-obj.forward(weighted-sum)
        return A, Z                     # return activations and weighted-sum of current-layer

    def backward(self, dA, A_prev, W, b, Z):    
        m = A_prev.shape[0]
        dZ = self.activation.backward(Z, dA)
        dW = 1 / m * np.dot(A_prev.T, dZ)
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, W.T)
        return dA_prev, dW, db

# Base-class interface
class LinearActivation:
    """
    Different activation functions both forward (given weighted sum) and backward (derivative)
    """
    def forward(self, Z):
        pass

    def backward(self, Z, dA=1):
        pass
class Linear(LinearActivation):
    def forward(self, Z):
        return Z

    def backward(self, Z, dA=1):
        return dA

class Sigmoid(LinearActivation):
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = s * (1 - s)
        return dA * dZ

class ReLU(LinearActivation):
    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ

class Initializers:
    """
    Given the previous layer size and current layer size
    return a specific initlized weight matrix using formulsa
    """
    
    @staticmethod
    def normal(prev_layer_size, layer_size):
        return np.random.randn(prev_layer_size, layer_size)

    @staticmethod
    def uniform(prev_layer_size, layer_size):
        return np.random.uniform(-1, 1, (prev_layer_size, layer_size))

    @staticmethod
    def glorot_normal(prev_layer_size, layer_size):
        return np.random.randn(prev_layer_size, layer_size) * np.sqrt(2 / (layer_size + prev_layer_size))

    @staticmethod
    def glorot_uniform(prev_layer_size, layer_size):
        limit = np.sqrt(6 / (layer_size + prev_layer_size))
        return np.random.uniform(-limit, limit, (prev_layer_size, layer_size))

    @staticmethod
    def he_normal(prev_layer_size, layer_size):
        return np.random.randn(prev_layer_size, layer_size) * np.sqrt(2 / prev_layer_size)

    @staticmethod
    def he_uniform(prev_layer_size, layer_size):
        limit = np.sqrt(6 / prev_layer_size)
        return np.random.uniform(-limit, limit, (prev_layer_size, layer_size))

class Loss:
    """
    Each cost function has a class which contains a forward method which computes the cost given predictions and labels. 
    Backward method is the derivative of that cost function. 
    """

    class BinaryCrossEntropy:

        @staticmethod
        def forward(AL, Y):
            return np.squeeze(-1 / Y.shape[0] * np.sum(np.dot(np.log(AL.T), Y) + np.dot(np.log(1 - AL.T), 1 - Y)))

        @staticmethod
        def backward(AL, Y):
            return -Y / AL + (1 - Y) / (1 - AL)

    class CategoricalCrossEntropy:

        @staticmethod
        def forward(AL, Y):
            return np.squeeze(-1 / Y.shape[0] * np.sum(Y * np.log(AL)))

        @staticmethod
        def backward(AL, Y):
            return -Y / AL

    class MSE:

        @staticmethod
        def forward(AL, Y):
            return np.squeeze(1 / Y.shape[0] * np.sum(np.square((Y - AL))))

        @staticmethod
        def backward(AL, Y):
            return -2 * (Y - AL)

class Optimizers:
    """
    Each class is a different optimizer. Which has a update method that updates/returns the parameters.
    """
    class SGD:  # stochastic gradient descent
        def __init__(self, learning_rate=0.001):
            self.learning_rate = learning_rate
            self.name = "SGD"

        def configure(self, W, b, layers):
            return W, b

        def update(self, W, b, layers, grad):
            for layer in range(len(layers)):        # iterate through each layer
                
                W[layer] -= self.learning_rate * grad[layer]['dW']      # update weights of cur-layer using learning-rate and gradient of weights/biases
                b[layer] -= self.learning_rate * grad[layer]['db']      # grad is dictionary

            return W, b
        
    class Momentum:
        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta
            self.name = "Momentum"

        def configure(self, W, b, VdW, Vdb, layers):
            VdW = []        
            Vdb = []
            for layer in range(len(layers)):        # iterate thourgh eachlayer and initalize veloctiy of cur-layer to that same shape as parameters of cur-layer
                VdW.append(np.zeros(W[layer].shape))
                Vdb.append(np.zeros(b[layer].shape))
            return W, b, VdW, Vdb

        def update(self, W, b, VdW, Vdb, layers, grad):
            for layer in range(len(layers)):        # iterate each layer-indx and update velocites using beta-HPP
                VdW[layer] = self.beta * VdW[layer] + (1 - self.beta) * grad[layer]["dW"]  
                Vdb[layer] = self.beta * Vdb[layer] + (1 - self.beta) * grad[layer]["db"]  

                W[layer] -= self.learning_rate * VdW[layer]
                b[layer] -= self.learning_rate * Vdb[layer]

            return W, b, VdW, Vdb
        
    class RMS_Prop:

        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta

        def configure(self, W, b, SdW, Sdb, layers):
            SdW = []
            Sdb = []
            for layer_indx in range(layers):
                SdW.append(np.zeros(W[layer_indx].shape))
                Sdb.append(np.zeros(W[layer_indx].shape))

        def update(self, W, b, SdW, Sdb, layers, grad):
            for layer_indx in range(len(layers)):
                SdW[layer_indx] = self.beta*SdW[layer_indx] + (1-self.beta)*((grad[layer_indx]["dW"])**2)
                Sdb[layer_indx] = self.beta*Sdb[layer_indx] + (1-self.beta)*((grad[layer_indx]["db"])**2)

                W[layer_indx] -= self.leanring_rate * (grad[layer_indx]["dW"])



class NeuralNetwork:

    def __init__(self):
        self.layers = []  # array of layer objects
        # self.parameters = []  # params = [{W:matrix, b:matrix}, {}], index each dictionary by layer-index and then its keys, layer -> W -> matrix
        self.W = []  # each element is a matrix for that index-layer
        self.b = []
        self.VdW = []   # stores velocties for each layer
        self.Vdb = []
        self.SdW = []
        self.Sdb = []
        self.caches = []   # params = [{dW:matrix, db:matrix}, {}], Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = []

    def add(self, layer):
        self.layers.append(layer)

    def setup(self, cost_func, input_size, optimizer):
        self.cost_func = cost_func  # set cost function 
        self.optimizer = optimizer  # set optimizer function
        self.initialize_weights_biases(input_size)

        # check for all optimization methods and initlize parameters
        if self.optimizer.name == "SGD": 
            self.W, self.b = self.optimizer.configure(self.W, self.b, self.layers)  # add some stuff to parameters
        if self.optimizer.name == "Momentum":
            self.W, self.b, self.VdW, self.Vdb = self.optimizer.configure(self.W, self.b, self.VdW, self.Vdb, self.layers)  # add some stuff to parameters
        

    def train(self, X, Y, epochs, learning_rate=0.001, batch_size=None, print_cost=True):
        
        self.optimizer.learning_rate = learning_rate

        num_examples = X.shape[0]       # number of examples
        

        for i in range(1, epochs + 1):
            x_batches = np.array_split(X, num_examples // batch_size, axis=0)
            y_batches = np.array_split(Y, num_examples // batch_size, axis=0)

            for x, y in zip(x_batches, y_batches):
                
                AL = self.forward(x)                  # feed x-data through network

                cost = self.cost_func.forward(AL, y)  # compute cost

                grad = self.backward(AL, y)  # get gradients, grad = [{'dW': dW, 'db': db}, {}, {}], each dictionary is for a layer

                if self.optimizer.name == "SGD":
                    self.W, self.b = self.optimizer.update(self.W, self.b, self.layers, grad)
                if self.optimizer.name == "Momentum":
                    self.W, self.b, self.VdW, self.Vdb = self.optimizer.update(self.W, self.b, self.VdW, self.Vdb, self.layers, grad)
                
                self.costs.append(cost)

            if print_cost and i%100 == 0:
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}")

    def initialize_weights_biases(self, input_size):
        layer_sizes = [input_size]
        for layer in self.layers:
            layer_sizes.append(layer.num_nodes)

        self.W, self.b = [], []
        for layer_indx in range(len(layer_sizes) - 1):
            W_layer, b_layer = initialize_parameters(layer_sizes[layer_indx], layer_sizes[layer_indx + 1], self.layers[layer_indx].initializer)
            self.W.append(W_layer)
            self.b.append(b_layer)


    def forward(self, A): # given activations which are x-inputs
        self.caches = []
        # iterate through each layer-indx
        for layer in range(len(self.layers)):
            A_prev = A  # store cur-activation as previous-activataion
            # layers-arr[layer-indx] call forward function passing in activations and parameters of cur-layer
            A, Z = self.layers[layer].forward(A_prev, self.W[layer], self.b[layer])
            # store previous-activations, weights, biases, and weighted-sum in cache
            self.caches.append({'A_prev': A_prev,"W": self.W[layer], "b": self.b[layer], "Z": Z})

        return A

    def backward(self, AL, Y):
        grad = []
        for _ in range(len(self.layers)):
            grad.append(0)

        # derivative of prev-layer activations given prediction-matrix and reshaped labes
        dA_prev = self.cost_func.backward(AL, Y.reshape(AL.shape))

        # iterate thorugh layers backward
        for layer_indx in reversed(range(len(self.layers))):
            cache = self.caches[layer_indx]  # current cache of layer 

            dA_prev, dW, db = self.layers[layer_indx].backward(dA_prev, **cache)

            grad[layer_indx] = {'dW': dW, 'db': db}

        return grad

    def predict(self, X):
        return self.forward(X)



if __name__ == "__main__":

    
    # SINE CURVE FITTING TASK
    def generate_noisy_sine_data(num_samples):
        X = np.linspace(0, 2*np.pi, num_samples)
        Y = np.sin(X) + 0.1 * np.random.randn(num_samples)
        return X, Y

    num_samples = 500
    X_train, Y_train = generate_noisy_sine_data(num_samples)

    X_train = X_train.reshape(-1, 1)


    model = NeuralNetwork()
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))

    model.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.Momentum(learning_rate=0.01))

    model.train(X_train, Y_train, epochs=5000, learning_rate=0.01, batch_size=num_samples)

    Y_pred = model.predict(X_train)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, Y_train, label='Noisy Data', color='blue')
    plt.plot(X_train, Y_pred, label='Predicted Curve', color='red')
    plt.title('Fitting a Noisy Sine Curve with Neural Network')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    