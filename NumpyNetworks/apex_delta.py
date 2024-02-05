import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters as zeros
# Bad - leads to symmetry, many neurons learn same features
def zeros(prev_layer_size, layer_size):
    return {
        'W': np.zeros((prev_layer_size, layer_size)), # Zeros for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters as ones
# Bad - leads to symmetry, many neurons learn same features
def ones(prev_layer_size, layer_size):
    return {
        'W': np.ones((prev_layer_size, layer_size)), # Ones for weights
        'b': np.ones((1, layer_size)) # Ones for biases
    }

# Initialize parameters following normal random distribution
def normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(prev_layer_size, layer_size), # Random normal for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters following uniform random distribution
def uniform(prev_layer_size, layer_size):
    return {
        'W': np.random.uniform(-1, 1, (prev_layer_size, layer_size)), # Random uniform for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters following Glorot / Xavier normal distribution
def glorot_normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(prev_layer_size, layer_size) * np.sqrt(2 / (layer_size + prev_layer_size)), # Glorot normal for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters following Glorot / Xavier uniform distribution
def glorot_uniform(prev_layer_size, layer_size):
    limit = np.sqrt(6 / (layer_size + prev_layer_size))
    return {
        'W': np.random.uniform(-limit, limit, (prev_layer_size, layer_size)), # Glorot uniform for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters following He normal distribution
def he_normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(prev_layer_size, layer_size) * np.sqrt(2 / prev_layer_size), # He normal for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }

# Initialize parameters following He uniform distribution
def he_uniform(prev_layer_size, layer_size):
    limit = np.sqrt(6 / prev_layer_size)
    return {
        'W': np.random.uniform(-limit, limit, (prev_layer_size, layer_size)), # He uniform for weights
        'b': np.zeros((1, layer_size)) # Zeros for biases
    }





class Dense():
    def __init__(self, units, activation, initializer=glorot_uniform):
        self.trainable = True
        self.units = units
        self.activation = activation
        self.initializer = initializer

    # Calculate layer neuron activations
    def forward(self, A_prev, W, b):
        Z = np.dot(A_prev, W) + b # Compute Z
        A = self.activation.forward(Z) # Compute A using the given activation function

        return A, Z
    
    # Find derivative with respect to weights, biases, and activations for a particular layer
    def backward(self, dA, A_prev, W, b, Z):
        m = A_prev.shape[0]
        dZ = self.activation.backward(Z, dA) # Evaluate dZ using the derivative of activation function
        dW = 1 / m * np.dot(A_prev.T, dZ) # Calculate derivative with respect to weights
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True) # Calculate derivative with respect to biases
        dA_prev = np.dot(dZ, W.T) # Calculate derivative with respect to the activation of the previous layer

        return dA_prev, dW, db
    

# Used in regression output layers
class Linear:

    def __init__(self, k=1):
        self.k = k

    # kz
    def forward(self, Z):
        return self.k * Z

    # k
    def backward(self, Z, dA=1):
        dZ = self.k
        return dA * dZ

# Maps output to [0, 1] (probability)
# Used in binary classification output layers
class Sigmoid:

    def __init__(self, c=1):
        self.c = c

    # 1 / (1 + e^-cz)
    def forward(self, Z):
        return 1 / (1 + np.exp(-self.c * Z))
    
    # c * s(z) * (1 - s(z))
    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = self.c * s * (1 - s)
        return dA * dZ
    

class ReLU():

    def __init__(self):
        pass

    # max(0,z)
    def forward(self, Z):
        return np.maximum(0, Z)
    
    # 0 if z <= 0, 1 if z > 0
    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ



# Binary Cross Entropy - Binary classification
class BinaryCrossEntropy:

    # (-1 / m * sum(Yln(A) + (1 - Y)ln(1 - A)))
    def forward(self, AL, Y):
        return np.squeeze(-1 / Y.shape[0] * np.sum(np.dot(np.log(AL.T), Y) + np.dot(np.log(1 - AL.T), 1 - Y)))
    
    # (-Y/A + (1 - Y)/(1 - A))
    def backward(self, AL, Y):
        return -Y / AL + (1 - Y) / (1 - AL)

# Categorical Cross Entropy - Multiclass classification
class CategoricalCrossEntropy:

    # (-1 / m * sum(Yln(A)))
    def forward(self, AL, Y):
        return np.squeeze(-1 / Y.shape[0] * np.sum(Y * np.log(AL)))

    # -Y / A
    def backward(self, AL, Y):
        return -Y / AL

# Mean Squared Error - Regression
class MSE:

    # (1 / m * sum((Y - A)^2))
    def forward(self, AL, Y):
        return np.squeeze(1 / Y.shape[0] * np.sum(np.square((Y - AL))))
    
    # (-2 * (Y - A))
    def backward(self, AL, Y):
        return -2 * (Y - AL)
    



class SGD:

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def configure(self, parameters, layers):
        return parameters

    def update(self, parameters, layers, grad):    
        for layer in range(len(layers)):
                if layers[layer].trainable:
                    parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW'] # Update weights
                    parameters[layer]['b'] -= self.learning_rate * grad[layer]['db'] # Update biases

        return parameters
    


class Model:

    # Initialize model
    def __init__(self):
        self.layers = [] # Each item is a layer object
        self.parameters = [] # Each item is a dictionary with a 'W' and 'b' matrix for the weights and biases respectively
        self.caches = [] # Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = [] # Each item is the cost for an epoch
            
    # Add layer to model
    def add(self, layer):
        self.layers.append(layer)

    # Configure model settings
    def configure(self, cost_type, input_size, optimizer):
        self.cost_type = cost_type
        self.optimizer = optimizer

        self.initialize_parameters(input_size) # Initialize parameters
        self.parameters = self.optimizer.configure(self.parameters, self.layers) # Add velocities, etc. to parameters

    # Train model
    def train(self, X, Y, epochs, learning_rate=0.001, batch_size=None, verbose=False):
        num_prints = 10 if epochs >= 10 else epochs # Print progress 10 times, if possible
        self.optimizer.learning_rate = learning_rate # Set learning rate
        m = X.shape[0] # Number of training samples
        if not batch_size: batch_size = m # Default to batch GD

        # Loop through epochs
        for i in range(1, epochs + 1):
            # Shuffle data, split into batches
            X_batches = np.array_split(X, m // batch_size, axis=0)
            Y_batches = np.array_split(Y, m // batch_size, axis=0)

            # Loop through batches
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                AL = self.forward(X_batch) # Forward propagate
                cost = self.cost_type.forward(AL, Y_batch) # Calculate cost
                grad = self.backward(AL, Y_batch) # Calculate gradient
                self.parameters = self.optimizer.update(self.parameters, self.layers, grad) # Update weights and biases
                self.costs.append(cost) # Update costs list

            if verbose and (i % (epochs // num_prints) == 0 or i == epochs):
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}") # Optional, output progress

    # Initialize weights and biases
    def initialize_parameters(self, input_size):
        # Initialize parameters using initializer functions
        layer_sizes = [input_size] + [layer.units for layer in self.layers if layer.trainable]
        self.parameters = [
            self.layers[layer].initializer(
                layer_sizes[layer],
                layer_sizes[layer + 1]
            )
            for layer in range(len(layer_sizes) - 1)
        ]

        # For non-trainable layers, set parameters to empty arrays
        for layer in range(len(self.layers)):
            if not self.layers[layer].trainable:
                self.parameters.insert(layer, {'W': np.array([]), 'b': np.array([])})

    # Shuffle data
    def shuffle(self, X, Y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation, :], Y[permutation, :]

    # Forward propagate through model
    def forward(self, A, train=True):
        self.caches = []

        # Exclude non-trainable layers (like dropout) when not training
        layers = [i for i in range(len(self.layers)) if self.layers[i].trainable or train]
        
        # Loop through hidden layers, calculating activations
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

    # Find derivative with respect to each activation, weight, and bias
    def backward(self, AL, Y):
        grad = [None] * len(self.layers)
        dA_prev = self.cost_type.backward(AL, Y.reshape(AL.shape)) # Find derivative of cost with respect to final activation

        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, **cache)
            grad[layer] = {'dW': dW, 'db': db}
            
        return grad

    # Predict given input
    def predict(self, X):
        return self.forward(X, train=False)
    

"""
# SUM TASK
train_x = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8],
    [0.9, 0.1],
    [0.2, 0.3],
    [0.4, 0.5],
    [0.6, 0.7],
    [0.8, 0.9],
    [0.1, 0.2]
])
train_y = np.sum(train_x, axis=1, keepdims=True)


print("X: "+str(train_x))
print("Y: "+str(train_y))
model = Model()
model.add(Dense(units=20, activation=ReLU()))
model.add(Dense(units=7, activation=ReLU()))
model.add(Dense(units=5, activation=ReLU()))
model.add(Dense(units=1, activation=Sigmoid(), initializer=he_uniform))

model.configure(
    cost_type=MSE(),
    optimizer=SGD(learning_rate=0.01),
    input_size=2
)

model.train(
    train_x,
    train_y,
    epochs=10000,
    learning_rate=0.01,
    batch_size=10,
    verbose=True
)

sample_input = np.array([[0.2, 0.2]])
prediction = model.predict(sample_input)
print(f"Input: {sample_input} prediction: {prediction}")
"""

# FIT CURVE TASK
def generate_sine_data(num_samples):
    X = np.linspace(0, 2*np.pi, num_samples)
    Y = np.sin(X)
    return X, Y

num_samples = 100
X_train, Y_train = generate_sine_data(num_samples)

X_train = X_train.reshape(-1, 1)

model = Model()
model.add(Dense(units=20, activation=ReLU()))
model.add(Dense(units=10, activation=ReLU()))
model.add(Dense(units=1, activation=Linear(), initializer=normal))

model.configure(
    cost_type=MSE(),
    optimizer=SGD(learning_rate=0.01),
    input_size=1
)

model.train(X_train, Y_train, epochs=5000, learning_rate=0.01, batch_size=num_samples, verbose=True)

X_test = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
Y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, label='Actual Data', color='blue')
plt.plot(X_test, Y_pred, label='Predicted Curve', color='red')
plt.title('Fitting a Sine Curve with Neural Network')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# XOR FUNCTION
"""X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_xor = np.array([[0], [1], [1], [0]])

# Instantiate and configure the model
xor_model = Model()
xor_model.add(Dense(units=5, activation=ReLU()))
xor_model.add(Dense(units=1, activation=Sigmoid(), initializer=he_uniform))

xor_model.configure(
    cost_type=BinaryCrossEntropy(),
    optimizer=SGD(learning_rate=0.01),
    input_size=2
)

xor_model.train(X_xor, Y_xor, epochs=10000, learning_rate=0.01, batch_size=4, verbose=True)

Y_xor_pred = xor_model.predict(np.array([1, 1]))
sample_input = np.array([1, 1])
prediction = xor_model.predict(sample_input)
print(f"Input: {sample_input} prediction: {prediction}")"""



# COLOR CLASSIFICATION TASK
"""
input_data = np.array([
    [255, 0, 0],    # Red - 0
    [0, 255, 0],    # Green - 1
    [0, 0, 255],    # Blue - 2
    [255, 255, 255],  # White - 3
    [0, 0, 0]       # Black - 4
])

# Corresponding color labels
output_data = np.array([
    [1, 0, 0, 0, 0],  # Red
    [0, 1, 0, 0, 0],  # Green
    [0, 0, 1, 0, 0],  # Blue
    [0, 0, 0, 1, 0],  # White
    [0, 0, 0, 0, 1]   # Black
])


# Instantiate and configure the model
color_model = Model()
color_model.add(Dense(units=10, activation=ReLU()))
color_model.add(Dense(units=5, activation=Sigmoid(), initializer=he_uniform))

color_model.configure(
    cost_type=CategoricalCrossEntropy(),
    optimizer=SGD(learning_rate=0.01),
    input_size=3  # Input size is 3 for RGB values
)

# Train the model
color_model.train(input_data / 255.0, output_data, epochs=1000, learning_rate=0.01, batch_size=len(input_data), verbose=True)

# Predict using the trained model
sample_input = np.array([[250, 0, 0]])  # Sample RGB value to predict
predicted_probabilities = color_model.predict(sample_input)

# Get the predicted color label
predicted_label = list(["R","G","B","W","B"])[np.argmax(predicted_probabilities)]

# Display the results
print("Predicted Probabilities:", predicted_probabilities)
print("Predicted Color Label:", predicted_label)"""