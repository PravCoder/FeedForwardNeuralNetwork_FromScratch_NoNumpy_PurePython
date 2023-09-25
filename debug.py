import math
import random
import matplotlib.pyplot as plt
# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize network architecture and hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden and output layers
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.bias_hidden = [0] * self.hidden_size
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        self.bias_output = 0

    def forward(self, x):
        # Forward pass through the network
        self.hidden_input = [0] * self.hidden_size
        self.hidden_output = [0] * self.hidden_size

        # Calculate the input and output of the hidden layer
        for i in range(self.hidden_size):
            self.hidden_input[i] = sum(x[j] * self.weights_input_hidden[j][i] for j in range(self.input_size)) + self.bias_hidden[i]
            self.hidden_output[i] = sigmoid(self.hidden_input[i])

        # Calculate the input and output of the output layer
        self.output_input = sum(self.hidden_output[i] * self.weights_hidden_output[i] for i in range(self.hidden_size)) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward(self, x, target):
        # Backpropagation
        error = target - self.output

        # Compute gradients for the output layer
        output_delta = error * sigmoid_derivative(self.output)

        # Update weights and biases for the hidden-to-output layer
        for i in range(self.hidden_size):
            self.weights_hidden_output[i] += self.learning_rate * output_delta * self.hidden_output[i]
        self.bias_output += self.learning_rate * output_delta

        # Compute gradients for the hidden layer
        hidden_deltas = [0] * self.hidden_size
        for i in range(self.hidden_size):
            hidden_deltas[i] = output_delta * self.weights_hidden_output[i] * sigmoid_derivative(self.hidden_output[i])

        # Update weights and biases for the input-to-hidden layer
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_deltas[j] * x[i]
            self.bias_hidden[j] += self.learning_rate * hidden_deltas[j]

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                output = self.forward(x)
                self.backward(x, target)
                total_error += (target - output) ** 2
            avg_error = total_error / len(X)
            print(f"Epoch {epoch + 1}/{epochs}, Average Error: {avg_error:.6f}")

if __name__ == "__main__":
    # Generate some noisy sine wave data
    X = [[x / 10] for x in range(0, 100)]
    y = [math.sin(x[0]) + random.uniform(-0.1, 0.1) for x in X]

    # Create and train the neural network
    nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1)

    # Test the trained network on new data points
    test_X = [[x / 10] for x in range(0, 100)]
    predictions = [nn.forward(x) for x in test_X]

    # Generate some noisy sine wave data
    X = [[x / 10] for x in range(0, 100)]
    y = [math.sin(x[0]) + random.uniform(-0.1, 0.1) for x in X]

    # Create and train the neural network
    nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1)
    nn.train(X, y, epochs=10000)

    # Test the trained network on new data points
    test_X = [[x / 10] for x in range(0, 100)]
    predictions = [nn.forward(x) for x in test_X]

    # Plot the sine curve
    sine_x = [x[0] for x in X]
    plt.plot(sine_x, y, label="Sine Curve (Noisy)")

    # Plot the network's predictions
    plt.plot(sine_x, predictions, label="Network Predictions")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sine Curve vs. Network Predictions")
    plt.legend()
    plt.grid()
    plt.show()
