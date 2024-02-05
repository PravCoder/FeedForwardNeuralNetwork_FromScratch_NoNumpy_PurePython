import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(prev_layer_size, layer_size, initializer):
    W_layer = initializer(prev_layer_size, layer_size)
    b_layer = np.zeros((1, layer_size))
    W_layer, b_layer
    return W_layer, b_layer

class Layer:
    def __init__(self, num_nodes, activation, initializer):
        self.num_nodes = num_nodes
        self.activation = activation
        self.initializer = initializer

    def forward(self, A_prev, W, b):
        # print(A_prev)
        # print("\n")
        # print(W)
        Z = np.dot(A_prev, W) + b
        A = self.activation.forward(Z)
        return A, Z

    def backward(self, dA, A_prev, W, b, Z):
        m = A_prev.shape[0]
        dZ = self.activation.backward(Z, dA)
        dW = 1 / m * np.dot(A_prev.T, dZ)
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, W.T)
        return dA_prev, dW, db

# Base-class interface
class LinearActivation:
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

    class SGD:
        def __init__(self, learning_rate=0.001):
            self.learning_rate = learning_rate
            self.name = "SGD"

        def configure(self, W, b, layers):
            return W, b

        def update(self, W, b, layers, grad):
            for layer in range(len(layers)):
                
                W[layer] -= self.learning_rate * grad[layer]['dW']
                b[layer] -= self.learning_rate * grad[layer]['db']

            return W, b
        
    class Momentum:
        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta
            self.name = "Momentum"

        def configure(self, W, b, VdW, Vdb, layers):
            VdW = []
            Vdb = []
            for layer in range(len(layers)):
                VdW.append(np.zeros(W[layer].shape))
                Vdb.append(np.zeros(b[layer].shape))
            return W, b, VdW, Vdb

        def update(self, W, b, VdW, Vdb, layers, grad):
            for layer in range(len(layers)):
                VdW[layer] = self.beta * VdW[layer] + (1 - self.beta) * grad[layer]["dW"]  # fix: use VdW[layer] instead of W[layer]
                Vdb[layer] = self.beta * Vdb[layer] + (1 - self.beta) * grad[layer]["db"]  # fix: use Vdb[layer] instead of b[layer]

                W[layer] -= self.learning_rate * VdW[layer]
                b[layer] -= self.learning_rate * Vdb[layer]

            return W, b, VdW, Vdb

class NeuralNetwork:

    def __init__(self):
        self.layers = []  # array of layer objects
        # self.parameters = []  # params = [{W:matrix, b:matrix}, {}], index each dictionary by layer-index and then its keys, layer -> W -> matrix
        self.W = []  # each element is a matrix for that index-layer
        self.b = []
        self.VdW = []
        self.Vdb = []
        self.caches = []   # params = [{dW:matrix, db:matrix}, {}], Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = []

    def add(self, layer):
        self.layers.append(layer)

    def setup(self, cost_func, input_size, optimizer):
        self.cost_func = cost_func
        self.optimizer = optimizer
        self.initialize_weights_biases(input_size)

        self.W, self.b, self.VdW, self.Vdb = self.optimizer.configure(self.W, self.b, self.VdW, self.Vdb, self.layers)  # add some stuff to parameters
        

    def train(self, X, Y, epochs, learning_rate=0.001, batch_size=None, print_cost=True):
        
        self.optimizer.learning_rate = learning_rate

        num_examples = X.shape[0]
        

        for i in range(1, epochs + 1):
            x_batches = np.array_split(X, num_examples // batch_size, axis=0)
            y_batches = np.array_split(Y, num_examples // batch_size, axis=0)

            for x, y in zip(x_batches, y_batches):
                AL = self.forward(x)                  # feed x-data through network
                cost = self.cost_func.forward(AL, y)  # compute cost

                grad = self.backward(AL, y)  # get gradients, grad = [{'dW': dW, 'db': db}, {}, {}], each dictionary is for a layer

                # old = self.W

                self.W, self.b, self.VdW, self.Vdb = self.optimizer.update(self.W, self.b, self.VdW, self.Vdb, self.layers, grad)

                # print(f'old: {old[0]}')
                # print(f'new: {self.W[0]}')

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


    def forward(self, A):
        self.caches = []

        for layer in range(len(self.layers)):
            A_prev = A
            A, Z = self.layers[layer].forward(A_prev, self.W[layer], self.b[layer])

            self.caches.append({'A_prev': A_prev,"W": self.W[layer], "b": self.b[layer], "Z": Z})

        return A

    def backward(self, AL, Y):
        grad = []
        for _ in range(len(self.layers)):
            grad.append(0)

        dA_prev = self.cost_func.backward(AL, Y.reshape(AL.shape))

        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]

            dA_prev, dW, db = self.layers[layer].backward(dA_prev, **cache)

            grad[layer] = {'dW': dW, 'db': db}

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
    