import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Model:

    def __init__(self, X, Y, dimensions, activations, iterations=1000, learning_rate=0.1, loss_type="binary_cross_entropy"):
        self.X = X                  # each row is an example, each element in row-example corresponds to a input node value
        self.Y = Y                  # each row is an output example label, each element in output-row-label corresponds to a output nodes label value
        self.dimensions = dimensions  # [1, 5,5,5 1] number of nodes per layer including input to output layers
        self.activations = activations  # ["R", "R","R","R", "S"] defines the activation used for each layer via the index, every layer as activation func
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss_type = loss_type

        self.W = {}  # dict where each key is layer-index and its value is a matrix of weights for that index-layer
        self.b = {}  # dict where key is layer-indx and its value is a matrix of biases for that index-layer
        self.intermed = []  # list where each element is dict with intermediate values {layer_indx, A_prev, W, b, Z}
        self.dW = {}
        self.db = {}

    # **********Activation Functions**********
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
    def linear_forward(self, Z):
        return Z
    def linear_backward(self, Z, dA=1):
        return dA
    def Softmax_forward(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # e to the power of every z-value in layer, subtract max for numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # sum all exp-values of every z-value in layer
        return probabilities
    def Softmax_backward(self, Z, Y):
        A = self.Softmax_forward(Z)  
        dZ = A - Y 
        return dZ
    
    # **********Loss Functions**********
    # Binary Cross Entropy
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
    # Mean-Squared-Error
    def mse_forward(self, AL, Y):
        return np.mean(np.square(AL - Y))
    def mse_backward(self, AL, Y):
        return 2 * (AL - Y) / Y.shape[0]
    # Categorical Cross Entropy
    def categorical_cross_entropy_forward(self, AL, Y):
        epsilon = 1e-15  # to avoid log(0)
        AL = np.clip(AL, epsilon, 1 - epsilon)  # clip prediction
        loss = -np.sum(Y * np.log(AL)) / Y.shape[0]
        return loss
    def categorical_cross_entropy_backward(self, AL, Y):
        epsilon = 1e-15  # To avoid division by 0
        AL = np.clip(AL, epsilon, 1 - epsilon)  # clip predictions
        dAL = -Y / AL
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


    def forward_propagation(self):
        self.intermed = []
        A_prev = self.X

        for layer_indx in range(1, len(self.dimensions)):
            # store the input A_prev before computing Z
            cur_A_prev = A_prev
            
            # weighted-sum
            Z = np.dot(A_prev, self.W[layer_indx]) + self.b[layer_indx]
            
            # activation function
            if self.activations[layer_indx] == "L":
                A = self.linear_forward(Z)
            if self.activations[layer_indx] == "R":
                A = self.ReLU_forward(Z)
            if self.activations[layer_indx] == "S":
                A = self.Sigmoid_forward(Z)
            if self.activations[layer_indx] == "SM":
                A = self.Softmax_forward(Z)
                
            # store the ORIGINAL A_prev, not the new A
            self.intermed.append({
                "layer": layer_indx,
                "A_prev": cur_A_prev,  # Store the input A_prev
                "W": self.W[layer_indx],
                "b": self.b[layer_indx],
                "Z": Z
            })
            
            # update A_prev for next layer
            A_prev = A
        
        return A  # return activations of final layer
    
        
    def train(self):
        self.prepare_network()
        costs = []
        
        for iter_num in range(1, self.iterations + 1):
            # forward propagation
            AL = self.forward_propagation()
            
            # compute cost based on loss type
            if self.loss_type == "binary_cross_entropy":
                cost = self.binary_cross_entropy_forward(AL, self.Y)
            elif self.loss_type == "mse":
                cost = self.mse_forward(AL, self.Y)
            elif self.loss_type == "categorical_cross_entropy":
                cost = self.categorical_cross_entropy_forward(AL, self.Y)
            
            # backward propagation
            if self.loss_type == "binary_cross_entropy":
                dAL = self.binary_cross_entropy_backward(AL, self.Y)
            elif self.loss_type == "mse":
                dAL = self.mse_backward(AL, self.Y)
            elif self.loss_type == "categorical_cross_entropy":
                dAL = self.categorical_cross_entropy_backward(AL, self.Y)
                
            self.dW, self.db = self.backward_propagation(AL, self.Y, dAL)
            
            # update parameters
            self.W, self.b = self.optimizer_update()
            
            if iter_num % 100 == 0:  # Print every 100 iterations
                print(f"Cost #{iter_num} is {cost}")
            costs.append(cost)
            
            # Early stopping if cost is not changing
            # if len(costs) > 10 and np.abs(costs[-1] - costs[-2]) < 1e-10:
            #     print("Cost stopped changing. Training complete.")
            #     break
        
        return costs


    def backward_propagation(self, AL, Y, dAL):
        self.dW, self.db = self.reset_grads()
        dA_prev = dAL  # Use the passed dAL instead of computing it
        
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            cache = self.get_cache(layer_indx)
            A_prev, W, b, Z = cache["A_prev"], cache["W"], cache["b"], cache["Z"]
            dA_prev, dW, db = self.layer_backward(dA_prev, A_prev, W, b, Z, layer_indx)
            self.dW[layer_indx] = dW
            self.db[layer_indx] = db
        
        return self.dW, self.db

    def layer_backward(self, dA, A_prev, W, b, Z, layer_indx):
        m = A_prev.shape[0]  # number of examples
        
        # Calculate dZ based on activation function
        if self.activations[layer_indx] == "L":
            dZ = self.linear_backward(Z, dA)
        if self.activations[layer_indx] == "R":
            dZ = self.ReLU_backward(Z, dA) 
        if self.activations[layer_indx] == "S":
            dZ = self.Sigmoid_backward(Z, dA)
        if self.activations[layer_indx] == "SM":
            dZ = self.Softmax_backward(Z, self.Y)
        
        # Calculate gradients
        # A_prev should have correct shape from stored intermediate values
        dW = 1 / m * np.dot(A_prev.T, dZ)
        db = 1 / m * np.sum(dZ, axis=0)
        dA_prev = np.dot(dZ, W.T)
        
        # Validation checks
        expected_dW_shape = W.shape
        if dW.shape != expected_dW_shape:
            raise ValueError(f"Layer {layer_indx}: dW shape {dW.shape} doesn't match W shape {expected_dW_shape}")
        
        # print(f"layer_backward(): layer_indx={layer_indx}")
        # print(f"  A_prev shape: {A_prev.shape}")
        # print(f"  dZ shape: {dZ.shape}")
        # print(f"  W shape: {W.shape}")
        # print(f"  dW shape: {dW.shape}")
        # print(f"  dA_prev shape: {dA_prev.shape}")
        
        return dA_prev, dW, db
    
    def optimizer_update(self):
        # iterate from 1st-layer to last-layer because input-layer doesnt have weights to update
        for key, val in self.dW.items():
            pass
            #print(f"dW-{key=}, {val.shape=}")

        for layer_indx in range(1, len(self.dimensions)):
            #print(f"Layer {layer_indx}: W.shape = {self.W[layer_indx].shape}, dW.shape = {self.dW[layer_indx].shape}")
            self.W[layer_indx] -= self.learning_rate * self.dW[layer_indx]
            self.b[layer_indx] -= self.learning_rate * self.db[layer_indx]
        return self.W, self.b  # returning for precaution
    
    def reset_grads(self):
        for layer_indx in range(len(self.dimensions)-1, 0, -1):
            n = self.dimensions[layer_indx-1] # number of nodes in previous-layer
            m = self.dimensions[layer_indx]   # number of nodes in current-layer
            self.dW[layer_indx]  = np.zeros((n, m))
            # print(f"reset layer: {layer_indx} - {self.dW[layer_indx].shape}")
            self.db[layer_indx] = np.zeros(m)
        return self.dW, self.db

    def get_cache(self, layer_indx):
        # Add validation
        for cache in self.intermed:
            if cache["layer"] == layer_indx:
                # Validate shapes before returning
                W_shape = cache["W"].shape
                A_prev_shape = cache["A_prev"].shape
                if A_prev_shape[1] != W_shape[0]:
                    raise ValueError(f"Layer {layer_indx}: A_prev shape {A_prev_shape} incompatible with W shape {W_shape}")
                return cache
        raise ValueError(f"Cache not found for layer {layer_indx}")
    
    def predict(self, X_new):
        self.X = X_new
        X_og = X_new
        self.X = X_new
        predictions = self.forward_propagation()
        self.X = X_og
        return predictions
    
    """
    For regression accuracy is not the correct metric to use, instead use the 
    """
    def accuracy(self, X_new, Y_new): # y-new is 2d-array each is row is example, in each row is labels for each output node
        self.X = X_new
        self.Y = Y_new
        if self.loss_type == "binary_cross_entropy":
            preds = self.predict(X_new)  # get predictions, has same structure as y-new.
            preds_binary = (preds > 0.5).astype(int)  # goes through 2d-arr and convets probabilties to prediction 0/1 if its less/greater than preds
            correct = np.sum(preds_binary == Y_new)  # element-wise comparison, counts all rows-examples wehre the binary
            total = Y_new.shape[0]  # get total examples, number of rows
            print(f"Correct examples: {correct}/{total}")
            return (correct / total) * 100
        if self.loss_type == "categorical_cross_entropy":
            preds = self.predict(X_new) 
            # For multi-class classification, get the class index with the highest probability
            preds_classes = np.argmax(preds, axis=1)  # 1d-array where each element is index of output node with high problaity for each example
            true_classes = np.argmax(Y_new, axis=1)  # same with true-classes. 
            # count correct predictions
            correct = np.sum(preds_classes == true_classes) # compares 1D-lists where they are equal and counts it .
            total = Y_new.shape[0]   # get number of examples
            print(f"Correct examples: {correct}/{total}")
            return (correct / total) * 100
    

# APPLICATIONS
def generate_sine_data(num_points=1000, noise_factor=0.1):
    X = np.linspace(0, 2*np.pi, num_points)
    Y = np.sin(X) + 0.1 * np.random.randn(num_points)    
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    
    return X, Y

def sine_curve_example(num_points): # optimal: 500 points, 0.0 lr, 1 100 50 1 dims, 1500 iterations.
    #num_points=500
    X_train, Y_train = generate_sine_data(num_points=num_points, noise_factor=0.1)
    dims = [1, 100,50, 1]  
    acts = ["INPUT", "R", "R","L"]   # first string is for placeholder input, then define hidden activations, then output layer activations.
    net = Model(X_train, Y_train, dims, acts, 
           iterations=1500,
           learning_rate=0.01,
           loss_type="mse") 
    
    net.train()
    Y_pred = net.predict(X_train)
    print(f"{len(Y_pred)=}, {len(X_train)=}")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, Y_train, label="Noisy Data", color="blue")
    plt.plot(X_train, Y_pred, label="Predicted Curve", color="red")
    plt.title("Fitting a Noisy Sine Curve with Neural Network")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def xor_function_example():  # does better with simpliler topology still varying results due to random weight initialization
    #np.random.seed(42)
    X1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y1 = [[0], [1], [1], [0]]
    dims = [2, 2,5, 1]
    acts = ["INPUT","R","R", "S"]
    net = Model(np.array(X1), np.array(Y1), dims, acts, iterations=2000, learning_rate=0.1, loss_type="binary_cross_entropy")
    net.train()
    preds = net.predict(X1)
    accuracy = net.accuracy(np.array(X1), np.array(Y1))
    print(f"Net predictions: {preds} actual: {Y1}")
    print(f"Accuracy: {accuracy}")
    print(f"X-shape : {X1.shape}, Y-shape : {Y1.shape}")

def breast_cancer_example(): 
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = data.data
    Y = data.target.reshape(-1, 1)  # reshape to (examples, 1), for 1 output node binary classification
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
    
    dims = [X_train.shape[1], 10, 10, 10, 5, 1]  
    acts = ["INPUT", "R", "R", "R","R", "S"]     
    
    net = Model(X_train, Y_train, dims, acts, iterations=2000, learning_rate=0.1, loss_type="binary_cross_entropy")
    net.train()
    
    preds = net.predict(X_train) # 2d-arr where each element is a examples predictions
    preds_binary = (preds > 0.5).astype(int)  # convert probabilties into predictions 1 if greater than 0.5 and 0 if less than 0.5
    print(f"First 10 Predictions: {preds_binary[:10].flatten()}")
    print(f"First 10 Actual:      {Y_train[:10].flatten()}")
    accuracy = net.accuracy(X_train, Y_train)
    print(f"X-shape : {X_train.shape}, Y-shape : {Y_train.shape}")
    print(f"Accuracy: {accuracy}")

def iris_flower_example():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    iris = load_iris()
    X = iris.data  # Features
    Y = iris.target  # Labels reshaped as a column vector

    # standerdize features
    print(f"before standerdizing features: {X[:5]}")  # this was causing cost to be constant/nan gradient explosion values to big
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"after standerdizing features: {X[:5]}")

    # one-hot encode labels, 1 class is 1 all other classes 0 for every example. 
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y.reshape(-1,1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

    dims = [X_train.shape[1], 10, 10, 10, Y_train.shape[1]]  # Input -> Hidden Layers -> Output
    acts = ["INPUT", "R", "R", "R", "SM"]  # Activations: ReLU for hidden layers, Softmax for output

    net = Model(np.array(X_train), np.array(Y_train), dims, acts, iterations=500, learning_rate=0.075, loss_type="categorical_cross_entropy")
    net.train()
    preds = net.predict(X_train)
    print(f"{preds[:5]=}")
    print(f"{Y_train[:5]=}")
    predicted_classes = np.argmax(preds, axis=1)  # 1d-array where each element is index of output node with high problaity for each example
    true_classes = np.argmax(Y_train, axis=1)     # 1d-array where each element is index of output node with highest value one hot encoding so 1, each element in 1d-arr is index of true label 1 for each example

    accuracy = net.accuracy(X_train, Y_train)
    print(f"X-shape : {X_train.shape}, Y-shape : {Y_train.shape}, predicted_classes_num_examples={len(predicted_classes)}") # (examples, input features), (examples output nodes)
    print(f"Accuracy: {accuracy}%")


    # print couple predictions
    print("---Sample Predictions---:")
    for i in range(20):
        print(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")

def MNIST_example(): # takes 20+ mins to train
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # input is 28*28 pixels 784, output is 0-9 digits 10 output nodes. 
    import numpy as np
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data / 255.0  # normalize
    Y = mnist.target.astype(int).to_numpy()  # convert target to integers

    # one-hot encode labels, but before resapce to 1d-vector
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y.reshape(-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    # standerdize further
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    dims = [X_train.shape[1], 128, 64, Y_train.shape[1]]  # Larger hidden layers for MNIST
    acts = ["INPUT", "R", "R", "SM"]  # ReLU for hidden layers, Softmax for output

    net = Model(np.array(X_train),np.array(Y_train),dims,acts,iterations=1000,  learning_rate=0.01, loss_type="categorical_cross_entropy")
    net.train()

    preds = net.predict(X_train)
    predicted_classes = np.argmax(preds, axis=1)  # 1d-array where each element is index of output node with high problaity for each example
    true_classes = np.argmax(Y_train, axis=1)     # 1d-array where each element is index of output node with highest value one hot encoding so 1, each element in 1d-arr is index of true label 1 for each example

    accuracy = net.accuracy(X_train, Y_train)

    print(f"X-shape : {X_train.shape}, Y-shape : {Y_train.shape}, predicted_classes_num_examples={len(predicted_classes)}") # (examples, input features), (examples output nodes)
    print(f"Accuracy: {accuracy}%")

    # print couple predictions
    print("---Sample Predictions---:")
    for i in range(20):
        print(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")


if __name__ == "__main__":
    
    # Binary Classification Tasks
    # xor_function_example()
    # breast_cancer_example()

    # Regression Tasks
    # sine_curve_example(500)

    # Multiclass classification
    # iris_flower_example()
    MNIST_example()
    pass
    
    

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


MNIST: REUSLTS:
dims: [784, 128, 64, 10]
weights of layer-1: (784, 128), bias: (128,)
weights of layer-2: (128, 64), bias: (64,)
weights of layer-3: (64, 10), bias: (10,)
Cost #100 is 8.115813034557128
Cost #200 is 7.160840114407793
Cost #300 is 6.473767163897253
Cost #400 is 5.887431840376309
Cost #500 is 5.224533040076349
Cost #600 is 4.963438657293821
Cost #700 is 4.311941083132232
Cost #800 is 3.593084488301908
Cost #900 is 3.3204185281900775
Cost #1000 is 2.9953705194712796
Correct examples: 46175/63000
X-shape : (63000, 784), Y-shape : (63000, 10), predicted_classes_num_examples=63000
Accuracy: 73.29365079365078%
---Sample Predictions---:
Predicted: 5, True: 8
Predicted: 7, True: 7
Predicted: 4, True: 6
Predicted: 6, True: 5
Predicted: 4, True: 4
Predicted: 4, True: 4
Predicted: 2, True: 2
Predicted: 4, True: 1
Predicted: 0, True: 4
Predicted: 7, True: 7
Predicted: 4, True: 4
Predicted: 0, True: 0
Predicted: 1, True: 1
Predicted: 5, True: 8
Predicted: 2, True: 2
Predicted: 4, True: 4
Predicted: 5, True: 8
Predicted: 8, True: 9
Predicted: 4, True: 4
Predicted: 5, True: 6
"""