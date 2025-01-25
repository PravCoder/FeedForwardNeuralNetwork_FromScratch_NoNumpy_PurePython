import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(prev_layer_size, layer_size, initializer):
    return {
        'W': initializer(prev_layer_size, layer_size),
        'b': np.zeros((1, layer_size))
    }

class Layer:
    def __init__(self, num_nodes, activation, initializer):
        self.num_nodes = num_nodes
        self.activation = activation
        self.initializer = initializer

    def forward(self, A_prev, W, b):
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

        def configure(self, parameters, layers):
            return parameters

        def update(self, parameters, layers, grad):
            for layer in range(len(layers)):
                
                parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW']
                parameters[layer]['b'] -= self.learning_rate * grad[layer]['db']
            return parameters
        
    class Momentum:
        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta

        def configure(self, parameters, layers):
            for layer in range(len(layers)):
                parameters[layer]["VdW"] = np.zeros(parameters[layer]["W"].shape)
                parameters[layer]["Vdb"] = np.zeros(parameters[layer]["b"].shape)
            return parameters

        def update(self, parameters, layers, grad):
            for layer in range(len(layers)):
                parameters[layer]["VdW"] = self.beta * parameters[layer]["VdW"] + (1 - self.beta) * grad[layer]["dW"]  # update velocties
                parameters[layer]["Vdb"] = self.beta * parameters[layer]["Vdb"] + (1 - self.beta) * grad[layer]["db"]

                parameters[layer]["W"] -= self.learning_rate * parameters[layer]["VdW"]  # update weights
                parameters[layer]["b"] -= self.learning_rate * parameters[layer]["Vdb"]  

            return parameters

class NeuralNetwork:

    def __init__(self):
        self.layers = []  # array of layer objects
        self.parameters = []  # params = [{W:matrix, b:matrix}, {}], index each dictionary by layer-index and then its keys, layer -> W -> matrix
        self.caches = []   # params = [{dW:matrix, db:matrix}, {}], Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = []

    def add(self, layer):
        self.layers.append(layer)

    def setup(self, cost_func, input_size, optimizer):
        self.cost_func = cost_func
        self.optimizer = optimizer
        self.initialize_weights_biases(input_size)
        self.parameters = self.optimizer.configure(self.parameters, self.layers)  # add some stuff to parameters

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

                self.parameters = self.optimizer.update(self.parameters, self.layers, grad)
                self.costs.append(cost)

            if print_cost and i%100 == 0:
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}")

    def initialize_weights_biases(self, input_size):
        layer_sizes = [input_size]
        for layer in self.layers:
            layer_sizes.append(layer.num_nodes)

        self.parameters = []
        for layer in range(len(layer_sizes) - 1):
            self.parameters.append(initialize_parameters(layer_sizes[layer], layer_sizes[layer + 1], self.layers[layer].initializer))


    def forward(self, A):
        self.caches = []

        for layer in range(len(self.layers)):
            A_prev = A
            A, Z = self.layers[layer].forward(A_prev, self.parameters[layer]['W'], self.parameters[layer]['b'])

            self.caches.append({'A_prev': A_prev,"W": self.parameters[layer]['W'],"b": self.parameters[layer]['b'], "Z": Z})

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
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))     # initalizer drastically affects performance
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))

    model.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.SGD(learning_rate=0.01))

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
    

    """
    x = np.array([
        [255, 0, 0],    # Red - 0
        [0, 255, 0],    # Green - 1
        [0, 0, 255],    # Blue - 2
        [255, 255, 255],  # White - 3
        [0, 0, 0]       # Black - 4
    ])

    # Corresponding color labels
    y = np.array([
        [1, 0, 0, 0, 0],  # Red
        [0, 1, 0, 0, 0],  # Green
        [0, 0, 1, 0, 0],  # Blue
        [0, 0, 0, 1, 0],  # White
        [0, 0, 0, 0, 1]   # Black
    ])


    model = NeuralNetwork()
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=5, activation=Sigmoid(), initializer=Initializers.glorot_uniform))

    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=3, optimizer=Optimizers.SGD(learning_rate=0.01))

    model.train(x, y, epochs=5000, learning_rate=0.01, batch_size=5)

    example = np.array([[255, 0, 0]])
    Y_pred = model.predict(example)
    print(f'Example: {example} prediction: {Y_pred}')
    
"""




    """
    # IRIS TASK
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import accuracy_score

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define and configure the neural network model
    model = NeuralNetwork()
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=3, activation=Sigmoid(), initializer=Initializers.glorot_uniform))  # Output layer with 3 units for Iris classes
    model.setup(cost_func=Loss.CategoricalCrossEntropy(), optimizer=Optimizers.Momentum(learning_rate=0.01, beta=0.9), input_size=X_train.shape[1])

    # Train the model
    model.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=100)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Convert predictions to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # built-in accuracy score
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    """