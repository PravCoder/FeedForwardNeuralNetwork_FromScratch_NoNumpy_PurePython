import numpy as np

class Dense:
    def __init__(self, units, activation, initializer=None):
        self.trainable = True
        self.units = units
        self.activation = activation
        self.initializer = initializer

    def initialize_parameters(self, prev_layer_size, layer_size):
        if self.initializer is None:
            return {
                'W': np.random.randn(prev_layer_size, layer_size),
                'b': np.zeros((1, layer_size))
            }
        else:
            return self.initializer(prev_layer_size, layer_size)

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

class Model:
    def __init__(self):
        self.layers = [] 
        self.parameters = [] 
        self.caches = [] 
        self.costs = [] 
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def configure(self, cost_type, input_size, optimizer):
        self.cost_type = cost_type
        self.optimizer = optimizer

        self.initialize_parameters(input_size)
        self.parameters = self.optimizer.configure(self.parameters, self.layers)

    def train(self, X, Y, epochs, learning_rate=0.001, batch_size=None, verbose=False):
        num_prints = 10 if epochs >= 10 else epochs 
        self.optimizer.learning_rate = learning_rate 
        m = X.shape[0] 
        if not batch_size: 
            batch_size = m 

        for i in range(1, epochs + 1):
            X, Y = self.shuffle(X, Y)
            X_batches = np.array_split(X, m // batch_size, axis=0)
            Y_batches = np.array_split(Y, m // batch_size, axis=0)

            for X_batch, Y_batch in zip(X_batches, Y_batches):
                AL = self.forward(X_batch) 
                cost = self.cost_type.forward(AL, Y_batch) 
                grad = self.backward(AL, Y_batch) 
                self.parameters = self.optimizer.update(self.parameters, self.layers, grad) 
                self.costs.append(cost) 

            if verbose and (i % (epochs // num_prints) == 0 or i == epochs):
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}")

    def initialize_parameters(self, input_size):
        layer_sizes = [input_size] + [layer.units for layer in self.layers if layer.trainable]
        self.parameters = [
            self.layers[layer].initialize_parameters(
                layer_sizes[layer],
                layer_sizes[layer + 1]
            )
            for layer in range(len(layer_sizes) - 1)
        ]

        for layer in range(len(self.layers)):
            if not self.layers[layer].trainable:
                self.parameters.insert(layer, {'W': np.array([]), 'b': np.array([])})

    def shuffle(self, X, Y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation, :], Y[permutation, :]

    def forward(self, A, train=True):
        self.caches = []
        layers = [i for i in range(len(self.layers)) if self.layers[i].trainable or train]
        
        for layer in layers:
            A_prev = A
            A, Z = self.layers[layer].forward(A_prev, self.parameters[layer]['W'], self.parameters[layer]['b'])
            self.caches.append({
                'A_prev': A_prev,
                "W": self.parameters[layer]['W'],
                "b": self.parameters[layer]['b'],
                "Z": Z
            })

        return A

    def backward(self, AL, Y):
        grad = [None] * len(self.layers)
        dA_prev = self.cost_type.backward(AL, Y.reshape(AL.shape))

        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, **cache)
            grad[layer] = {'dW': dW, 'db': db}
            
        return grad

    def predict(self, X):
        return self.forward(X, train=False)




import numpy as np
import matplotlib.pyplot as plt

# Generate sine curve dataset
np.random.seed(42)
X = np.sort(6 * np.random.rand(1000, 1) - 3, axis=0)
Y = np.sin(X).ravel()

# Plot the sine curve
plt.scatter(X, Y, color='blue', label='Actual')
plt.title('Sine Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# Define the model
model = Model()
model.add(Dense(units=1, activation=Sigmoid(), initializer=glorot_uniform))
model.configure(cost_type=MSE(), input_size=1, optimizer=SGD(learning_rate=0.01))

# Train the model
model.train(X, Y.reshape(-1, 1), epochs=1000, batch_size=32, verbose=True)

# Plot the learned curve
predictions = model.predict(X)
plt.scatter(X, Y, color='blue', label='Actual')
plt.scatter(X, predictions, color='red', label='Predicted')
plt.title('Sine Curve Fitting')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
