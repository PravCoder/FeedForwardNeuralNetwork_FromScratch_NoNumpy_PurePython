import numpy as np
import matplotlib.pyplot as plt
import json


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
        m = A_prev.shape[0]  # number of examples is the length of first index of activation-prev-layer
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

    def backward(self, Z, dA=1): # derivative of linear-activation is just the derivative of activations. 
        return dA

class Sigmoid(LinearActivation):
    def forward(self, Z):
        Z = np.clip(Z, -500, 500)  
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = s * (1 - s)
        return dA * dZ
    
class Softmax(LinearActivation):

    def forward(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # e to the power of every z-value in layer, subtract max for numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # sum all exp-values of every z-value in layer
        return probabilities
    
    def backward(self, Z, dA=1):
        return dA

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
            epsilon = 1e-15  
            AL = np.clip(AL, epsilon, 1 - epsilon)  # clip activations of last-layer min=epsilon max=1-epsilon, if values in AL are less than min it becomes min if values are more than max it becomes max
            # compute loss which is average of all examples Y-shape(examples, output-nodes), perform element-wise multiplication
            loss = -1 / Y.shape[0] * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
            return loss 

        @staticmethod
        def backward(AL, Y):
            epsilon = 1e-15  
            AL = np.clip(AL, epsilon, 1 - epsilon)  
            dAL = - (Y / AL) + (1 - Y) / (1 - AL)   # derivative of 
            return dAL / Y.shape[0]

    class CategoricalCrossEntropy:

        @staticmethod
        def forward(AL, Y):
            epsilon = 1e-15  # to avoid log(0)
            AL = np.clip(AL, epsilon, 1 - epsilon)  # clip prediction
            # compute categorical loss formula, y-shape(examples, output-nodes)
            loss = -np.sum(Y * np.log(AL)) / Y.shape[0]
            return loss

        @staticmethod
        def backward(AL, Y):
            epsilon = 1e-15  # to avoid division by 0
            AL = np.clip(AL, epsilon, 1 - epsilon)  # clip prediction
            
            dAL = - (Y / AL) + (1 - Y) / (1 - AL)
            return dAL / Y.shape[0] 

    class MSE:

        @staticmethod
        def forward(AL, Y):
            # print(Y.shape[0])
            return np.squeeze(1 / Y.shape[0] * np.sum(np.square((Y - AL))))

        @staticmethod
        def backward(AL, Y):
            return -2 * (Y - AL)
        
    class RMSE:

        @staticmethod
        def forward(AL, Y):
            return np.sqrt(np.mean(np.square(Y - AL)))

        @staticmethod
        def backward(AL, Y):
            m = Y.shape[0]
            return (1 / m) * (AL - Y) / np.sqrt(np.mean(np.square(AL - Y)) + 1e-8)
        
        

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

        def update(self, W, b, dW, db, layers):
            for layer in range(len(layers)):        # iterate through each layer
                
                W[layer] -= self.learning_rate * dW[layer]     # update weights of cur-layer using learning-rate and gradient of weights/biases
                b[layer] -= self.learning_rate * db[layer]    # grad is dictionary

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

        def update(self, W, b, dW, db, VdW, Vdb, layers):
            for layer in range(len(layers)):        # iterate each layer-indx and update velocites using beta-HPP
                VdW[layer] = self.beta * VdW[layer] + (1 - self.beta) * dW[layer] 
                Vdb[layer] = self.beta * Vdb[layer] + (1 - self.beta) * db[layer]

                W[layer] -= self.learning_rate * VdW[layer]
                b[layer] -= self.learning_rate * Vdb[layer]

            return W, b, VdW, Vdb
        
    class RMS_Prop:

        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta
            self.name = "RMS_Prop"
            self.epsilon = 1e-9

        def configure(self, W, b, SdW, Sdb, layers):
            SdW = []
            Sdb = []
            for layer_indx in range(len(layers)):
                SdW.append(np.zeros(W[layer_indx].shape))
                Sdb.append(np.zeros(b[layer_indx].shape))
            return W, b, SdW, Sdb

        def update(self, W, b, dW, db, SdW, Sdb, layers):
            for layer_indx in range(len(layers)):
                SdW[layer_indx] = self.beta*SdW[layer_indx] + (1-self.beta)*(dW[layer_indx]**2)
                Sdb[layer_indx] = self.beta*Sdb[layer_indx] + (1-self.beta)*(db[layer_indx]**2)

                W[layer_indx] -= self.learning_rate * (dW[layer_indx] / np.sqrt(SdW[layer_indx] + self.epsilon))
                b[layer_indx] -= self.learning_rate * (db[layer_indx] / np.sqrt(Sdb[layer_indx] + self.epsilon))
            return W, b, SdW, Sdb
        
    class Adam:

        def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
            self.leanring_rate = learning_rate
            self.beta1 = beta1  # momentum beta
            self.beta2 = beta2  # RMS beta
            self.epsilon = 1e-9
            self.name = "Adam"

        def configure(self, W, b, VdW, SdW, Vdb, Sdb, layers):
            VdW = []
            Vdb = []
            SdW = []
            Sdb = []
            for layer_indx in range(len(layers)):
                VdW.append(np.zeros(W[layer_indx].shape))
                Vdb.append(np.zeros(b[layer_indx].shape))
                SdW.append(np.zeros(W[layer_indx].shape))
                Sdb.append(np.zeros(b[layer_indx].shape))
            return W, b, VdW, Vdb, SdW, Sdb

        def update(self, W, b, dW, db, VdW, SdW, Vdb, Sdb, layers):
            for layer_indx in range(len(layers)):
                VdW[layer_indx] = self.beta1 * VdW[layer_indx] + (1 - self.beta1) * dW[layer_indx] # Update weight velocities
                Vdb[layer_indx] = self.beta1 * Vdb[layer_indx] + (1 - self.beta1) * db[layer_indx] # Update bias velocities
                SdW[layer_indx] = self.beta2 * SdW[layer_indx] + (1 - self.beta2) * np.square(dW[layer_indx]) # Update weight LR scalers
                Sdb[layer_indx] = self.beta2 * Sdb[layer_indx] + (1 - self.beta2) * np.square(db[layer_indx]) # Update bias LR scalers
                
                W[layer_indx] -= self.learning_rate * VdW[layer_indx] / (np.sqrt(SdW[layer_indx]) + self.epsilon) # Update weights
                b[layer_indx] -= self.learning_rate * Vdb[layer_indx] / (np.sqrt(Sdb[layer_indx]) + self.epsilon) # Update biases
                
            return W, b, dW, db, VdW, SdW, Vdb, Sdb




class NeuralNetwork:

    def __init__(self):
        self.layers = []  # array of layer objects
        # self.parameters = []  # params = [{W:matrix, b:matrix}, {}], index each dictionary by layer-index and then its keys, layer -> W -> matrix
        self.W = []  # each element is a matrix for that index-layer
        self.b = []
        self.dW = []
        self.db = []
        self.VdW = []   # stores velocties for each layer
        self.Vdb = []
        self.SdW = []
        self.Sdb = []
        self.caches = []   # params = [{dW:matrix, db:matrix}, {}], Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = []
        self.input_size = None

    def add(self, layer):
        self.layers.append(layer)

    def setup(self, cost_func, input_size, optimizer, is_gan_model=""):
        self.input_size = input_size
        self.cost_func = cost_func  # set cost function 
        self.optimizer = optimizer  # set optimizer function
        self.initialize_weights_biases(input_size)
        self.is_gan_model = is_gan_model

        # check for all optimization methods and initlize parameters
        if self.optimizer.name == "SGD": 
            self.W, self.b = self.optimizer.configure(self.W, self.b, self.layers)  # add some stuff to parameters
        if self.optimizer.name == "Momentum":
            self.W, self.b, self.VdW, self.Vdb = self.optimizer.configure(self.W, self.b, self.VdW, self.Vdb, self.layers)  # add some stuff to parameters
        if self.optimizer.name == "RMS_Prop":
            self.W, self.b, self.SdW, self.Sdb = self.optimizer.configure(self.W, self.b, self.SdW, self.Sdb, self.layers)
        if self.optimizer.name == "Adam":
            self.W, self.b, self.VdW, self.Vdb, self.SdW, self.Sdb = self.optimizer.configure(self.W, self.b, self.VdW, self.SdW, self.Vdb, self.Sdb, self.layers)
        

    def train(self, X, Y, epochs, learning_rate=0.001, batch_size=None, print_cost=True, D_out_real=None, D_out_fake=None, Y_fake=None, input_type=""):
        
        self.optimizer.learning_rate = learning_rate
        self.input_type = input_type
        self.D_out_real = D_out_real
        self.D_out_fake = D_out_fake
        self.Y_fake = Y_fake

        num_examples = X.shape[0]       # number of examples
        
        # MAIN LOOP
        for i in range(1, epochs + 1):
            x_batches = np.array_split(X, num_examples // batch_size, axis=0)
            y_batches = np.array_split(Y, num_examples // batch_size, axis=0)

            for x, y in zip(x_batches, y_batches):
                
                AL = self.forward(x)                  # feed x-data through network
                if self.is_gan_model == "":
                    cost = self.cost_func.forward(AL, y)  # compute cost
                if self.is_gan_model == "G":
                    cost = self.cost_func.forward(AL, Y, self.D_out_fake)
                if self.is_gan_model == "D":
                    cost = self.cost_func.forward(AL, Y, self.D_out_real, self.D_out_fake)

                self.backward(AL, y)  # get gradients, grad = [{'dW': dW, 'db': db}, {}, {}], each dictionary is for a layer

                if self.optimizer.name == "SGD":
                    self.W, self.b = self.optimizer.update(self.W, self.b, self.dW, self.db, self.layers)
                if self.optimizer.name == "Momentum":
                    self.W, self.b, self.VdW, self.Vdb = self.optimizer.update(self.W, self.b, self.dW, self.db, self.VdW, self.Vdb, self.layers)
                if self.optimizer.name == "RMS_Prop":
                    self.W, self.b, self.SdW, self.Sdb = self.optimizer.update(self.W, self.b, self.dW, self.db, self.SdW, self.Sdb, self.layers)
                if self.optimizer.name == "Adam":
                    self.W, self.b, self.dW, self.db, self.VdW, self.SdW, self.Vdb, self.Sdb = self.optimizer.update(self.W, self.b, self.dW, self.db, self.VdW, self.SdW, self.Vdb, self.Sdb, self.layers)
                

                self.costs.append(cost)

            if print_cost and i%100 == 0:
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}")

    def initialize_weights_biases(self, input_size):
        layer_sizes = [input_size]  # make sure initial size is there because there is no layer-object for input-layer
        for layer in self.layers:
            layer_sizes.append(layer.num_nodes)

        self.W, self.b = [], []
        for layer_indx in range(len(layer_sizes) - 1):  # for every layer-indx excluding input-layer
            W_layer, b_layer = initialize_parameters(layer_sizes[layer_indx], layer_sizes[layer_indx + 1], self.layers[layer_indx].initializer)
            self.W.append(W_layer)
            self.b.append(b_layer)


    def forward(self, A): # given activations which are x-inputs
        self.caches = []
        # iterate through each layer-indx
        for layer in range(len(self.layers)):
            A_prev = A  # store cur-activation as previous-activataion
            # layers-arr[layer-indx] call forward function passing in activations and parameters of cur-layer, returns activation-matrix and weight-sum-matrix
            A, Z = self.layers[layer].forward(A_prev, self.W[layer], self.b[layer])
            # store previous-activations, weights, biases, and weighted-sum in cache
            self.caches.append({'A_prev': A_prev,"W": self.W[layer], "b": self.b[layer], "Z": Z})

        return A

    def backward(self, AL, Y):
        self.dW, self.db = [], []       # reset and initalize grads after every iteration 0 for every grad matrix
        for _ in range(len(self.layers)):
            self.dW.append(0)
            self.db.append(0)

        # derivative of prev-layer activations given prediction-matrix and reshaped labes
        dA_prev = 0
        if self.is_gan_model == "":
            dA_prev = self.cost_func.backward(AL, Y.reshape(AL.shape))
        if self.input_type == "real":
            dA_prev = self.cost_func.backward(Y, self.D_out_real, self.D_out_fake, self.Y_fake, input_type="real")
        if self.input_type == "fake":
            dA_prev = self.cost_func.backward(Y, self.D_out_real, self.D_out_fake, self.Y_fake, input_type="fake")

        # iterate thorugh layers backward
        for layer_indx in reversed(range(len(self.layers))):
            cache = self.caches[layer_indx]  # current cache of layer 

            dA_prev, dW, db = self.layers[layer_indx].backward(dA_prev, **cache)

            self.dW[layer_indx] = dW # set the graidnets of cur-layer equal to matrix
            self.db[layer_indx] = db

        return None

    def predict(self, X):
        return self.forward(X)
    
    def evaluate_accuracy(self, X, Y_hat):
        pass

    # SAVE LOAD MODEL FUNCTIONS
    def save(self, file_path):
        json_params = [] # [{W-layer-1:[], b-layer-1:[]}]
        for layer_indx in range(len(self.layers)):  # iterate through every layer-indx with 0th-layer being the 1st-hiddeen-layer
            json_params.append({"W":self.W[layer_indx].tolist(), "b":self.b[layer_indx].tolist()})  # add a dictionary of weights/biases of cur-layer, convert to list first

        json_object = json.dumps(json_params, indent=4)
        with open(file_path, "w") as outfile: # open json-file and write the json-object
            outfile.write(json_object)

    def load(self, file_path):
        with open(file_path, 'r') as file:
            json_params = json.load(file)  # load the json-parameters
            djsoned_W = [] 
            djsoned_b = []
            for layer_indx in range(len(self.layers)):
                cur_W = np.array(json_params[layer_indx]["W"])
                cur_b = np.array(json_params[layer_indx]["b"])
                djsoned_W.append(cur_W)
                djsoned_b.append(cur_b)
        
        self.W = djsoned_W
        self.b = djsoned_b

    def print_network_architecture(self):
        print(f"\nNetwork Architecture: loss={self.cost_func}")
        print(f"*Layer({0}): nodes={self.input_size} act=None")
        for i, layer in enumerate(self.layers):
            if i != len(self.layers)-1:
                print(f"Layer({i+1}): nodes={layer.num_nodes} act={type(layer.activation).__name__}")
            else:
                print(f"*Layer({i+1}): nodes={layer.num_nodes} act={type(layer.activation).__name__}")
                
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
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform)) # this is the 1st layer not input-layer, input-layer does not have activation
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform)) # this is output-layer activation, can be softmax, sigmoid, tanh
    # number of input nodes is specified here
    model.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.Adam(learning_rate=0.01))

    model.train(X_train, Y_train, epochs=1000, learning_rate=0.01, batch_size=num_samples, print_cost=True)
    # # model.save("NeuralNetworkFromScratch/sample.json")
    # # model.load("NeuralNetworkFromScratch/sample.json")

    Y_pred = model.predict(X_train)  # [e1, e2, e3, e4], e4 = [n1, n2, n3]
    print(Y_pred[0])  # ith example
    print(X_train.shape)
    print(Y_train.shape)


    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, Y_train, label='Noisy Data', color='blue')
    plt.plot(X_train, Y_pred, label='Predicted Curve', color='red')
    plt.title('Fitting a Noisy Sine Curve with Neural Network')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    model.print_network_architecture()
    
    print(Y_train)
    print(Y_train.shape)


"""
TODO:
[X] GAN
[X] Binary classification
[X] Multiclass classification
[-] MNIST (not performing well)
[] Image classification
[] RNN
[] Image classification
[] Gradient checking
[] Droput
[] Learing rate decay
[] L2-regularization
"""