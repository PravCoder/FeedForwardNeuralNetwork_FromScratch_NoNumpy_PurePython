import numpy as np
import matplotlib.pyplot as plt


class FeedForwardNN:

    def __init__(self, train_X, train_Y, layers_dims, learning_rate, num_iterations):
        self.train_X = train_X
        self.train_Y = train_Y
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}  # {W1:matrix, b1:matrix}
        self.grads = {}  # {dA1:matrix, db1:matrix}
        self.caches = []
        self.AL = 0   # A[l] = activations-last-layer = y-hat = predictions
        self.m = train_Y.shape[1]
        self.initialize_parameters()
        self.costs = []

    def gradient_checking(self, epsilon=1e-7):
        # Set epsilon value for small perturbation

        # Initialize empty dictionary for gradients
        gradients = {}

        # Perform gradient checking for each parameter
        for param_key in self.parameters:
            # Create a copy of the parameter values
            original_param = np.copy(self.parameters[param_key])

            # Compute the gradients using backpropagation
            self.feed_forward()
            self.compute_cost()
            self.backpropagate()

            # Iterate over each element in the parameter matrix
            for i in range(original_param.shape[0]):
                for j in range(original_param.shape[1]):
                    # Compute the numerical gradient using forward and backward perturbation
                    self.parameters[param_key][i, j] = original_param[i, j] + epsilon
                    self.feed_forward()
                    self.compute_cost()
                    cost_plus = self.cost

                    self.parameters[param_key][i, j] = original_param[i, j] - epsilon
                    self.feed_forward()
                    self.compute_cost()
                    cost_minus = self.cost

                    numerical_gradient = (cost_plus - cost_minus) / (2 * epsilon)

                    # Reset the parameter value
                    self.parameters[param_key][i, j] = original_param[i, j]

                    # Store the numerical gradient in the gradients dictionary
                    gradients[(param_key, i, j)] = numerical_gradient

            # Compute the relative difference between the numerical and backpropagation gradients
            backprop_gradients = self.grads
            for key in gradients:
                numerator = np.abs(backprop_gradients[key] - gradients[key])
                denominator = np.abs(backprop_gradients[key]) + np.abs(gradients[key])
                difference = numerator / denominator

                if difference > 1e-7:
                    print("Gradient checking failed for parameter", key)
                    print("Backpropagation gradient:", backprop_gradients[key])
                    print("Numerical gradient:", gradients[key])
                    print("Difference:", difference)
                else:
                    print("Gradient checking passed for parameter", key)


    def initialize_parameters(self):
        np.random.seed(3)
        L = len(self.layers_dims) # number of layers
        # loop from 1st-hidden to output-layer inclusive. Through all hidden-layers
        for l in range(1, L): 
            # weights-layer-l = shape(n[L], n[l-1])
            self.parameters["W"+str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) *0.01
            # bias-layer-l = shape(n[l], 1)
            self.parameters["b"+str(l)] = np.zeros((self.layers_dims[l], 1))

    def sigmoid(self, Z):     # given Z of shape(n[l], m)
        A = 1/(1+np.exp(-Z)) # activation-layer-l = sigmoid(Z) of shape(n[l], m)
        cache = Z            # return activation-layer-l, cache(Z). Cache is what was used to compute activation-layer-l
        return A, cache     
    def relu(self, Z):    # given Z of shape(n[l], m)
        A = np.maximum(0, Z) # activation-layer-l = max(0, Z) of shape(n[l], m)
        cache = Z           # return activation-layer-l, cache(Z). Cache is what was used to compute activation-layer-l
        return A, cache

    def linear_forward(self, A, W, b):  # given activation-layer-l, weights-layer-l, bias-layer-l
        """print("W: " + str(W.shape))
        print("A: " + str(A.shape))
        print("-------")"""
        Z = np.dot(W, A) + b  # Z-layer-l = activation-layer-l * weights-layer-l + bias-layer-l
        cache = (A,W,b)       # return Z-layer-l, cache(A,W,b). Cache is what was used to compute Z-layer-l
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation): # given activations-layer-[l-1], weights-layer-l, 'activation-type'
        Z, linear_cache = self.linear_forward(A_prev, W, b) # returns Z-layer-l and linear-cache(A,W,b)
        if activation ==  "sigmoid": # if activation-type is sigmoid pass Z-layer-l into sigmoid-func()
            A, activation_cache = self.sigmoid(Z)
        if activation == "relu": # if activaiton-type is relu pass Z-layer-l into relu-func()
            A, activation_cache = self.relu(Z)
        # linear-cache is what was used to compute Z. activation-cache is what was used to compute A
        cache = (linear_cache, activation_cache) 
        return A, cache

    def feed_forward(self): # Forward-Propagation-Model
        self.caches = [] # stores cache(linear-cache, activation-cache) = ((A_prev,W,b), (Z)) for each layer
        L = len(self.parameters) // 2 # last index of all layers
        A = self.train_X  # A[0] = X.
        # loop from layer-1 to output-layer. Through all hidden-layers
        for  l in range(1, L):
            A_prev = A  # A[l-1] activations for previous layer = A which was computed in previous iteration
            # A[l], (linear, activation) = call linear-activation-forward() passing W[l] and b[l] and relu-activation-type because hidden-layer
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W"+str(l)], self.parameters["b"+str(l)], "relu")
            self.caches.append(cache) # add cache(linear-cache, activation-cache) to caches-list

        # A[L], (linear,activation) = call linear-activation-forward() passing W[L] and b[L] and sigmoid-activation-type because last-layer
        self.AL, cache = self.linear_activation_forward(A, self.parameters["W"+str(L)], self.parameters["b"+str(L)], "sigmoid")
        self.caches.append(cache) # add cache(linear-cache, activation-cache) to caches-list

    def compute_cost(self):
        self.m = self.train_Y.shape[1] # num of examples = Y of shape(n[L], m)
        # calculate cost usinf cross-entropy and Y-labels-matrix and A[L]-activations-layer-L-predictions
        self.cost = -1/self.m * np.sum(np.multiply(self.train_Y, np.log(self.AL)) + np.multiply((1-self.train_Y), np.log(1-self.AL)))
        self.cost = np.squeeze(self.cost)

    def linear_backward(self, dZ, cache): # given dZ[l] shape(n[l], m) and cache(A[l-1], W[l], b[l])
        A_prev, W, b = cache
        self.m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T)/self.m # compute dW[l] using dZ[l] and A_prev[l]
        db = np.sum(dZ, axis=1, keepdims=True) /self.m # compute db[l] using dZ[l]
        dA_prev = np.dot(W.T, dZ) # compute dA[l-1] using W[l] and dZ[l]
        return dA_prev, dW, db  # return gradients of dA[l],dW[l],db[l] respect to cur-cost

    def relu_backward(self, dA, cache): # given dA[l] and Z[l] compute dZ, relu-derivative
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ = np.array(Z > 0, dtype='float')
        return dZ
    def sigmoid_backward(self, dA, cache):  # given dA[l] and Z[l] compute dZ, sigmoid-derivative
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ 

    def linear_activation_backward(self, dA, cache, activation): # given dA[l],cache,activation-type
        linear_cache, activation_cache = cache # (A_prev[l],W[l],b[l]), (Z[l])
        if activation == "relu":
            # dZ[l] = call relu-derivative() passing dA[l] and Z[l]
            dZ = self.relu_backward(dA, activation_cache)
        if activation == "sigmoid":
            # dZ[l] = call sigmoid-derivative() passing dA[l] and Z[l]
            dZ = self.sigmoid_backward(dA, activation_cache)

        # dA[l-1] and dW[l] and db[l] = call linear-derivative() computes gradients given dZ[l] and (A_prev[l],W[l],b[l])
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backpropagate(self):
        L = len(self.caches) # last index of all layers
        self.train_Y = self.train_Y.reshape(self.AL.shape) # reshape Y into A[L]-shape(n[L],m)
        # compute dA[L] using Y-labels adn A[L]-predictions which are the derivatices of predictions respect to loss
        dAL = - (np.divide(self.train_Y, self.AL) - np.divide(1 - self.train_Y, 1 - self.AL)) # compute dA[4]
        cur_cache = self.caches[L-1] # get cache of dA[3]

        # compute gradients dA[L-1],dW[L],db[L] using dA[L] and (linear-cache, activation-cache). This backprogates last hidden layer
        dA_prev,dW_temp,db_temp = self.linear_activation_backward(dAL,cur_cache, "sigmoid")
        self.grads["dA"+str(L-1)] = dA_prev # set dA[3], dW[4], db[4] using dA[4]
        self.grads["dW"+str(L)] = dW_temp
        self.grads["db"+str(L)] = db_temp
        # loop from layer indicies 2->0 inclusive
        for l in reversed(range(L-1)):
            cur_cache = self.caches[l] # get cur-cache(linear[l], activation[l])
            dA_prev_temp, dW_temp,db_temp = self.linear_activation_backward(dA_prev, cur_cache,"relu")
            self.grads["dA" + str(l)] = dA_prev_temp  # set dA[l] is after layer-l
            self.grads["dW"+ str(l+1)] = dW_temp     # set dW[l+1] is before layer-l+1 and after layer-l
            self.grads["db" + str(l+1)] = db_temp   # 

    def update_parameters(self):
        L = len(self.parameters)//2
        # loop from input-layerindx-0 to last-hidden-layer inclusive
        for l in range(L): # range(1, L): str(l) also works
            # update parameters using gradients of cur-layer-indx. Using +1 because looping from zero indx
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - (self.learning_rate * self.grads["dW" + str(l + 1)])
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - (self.learning_rate * self.grads["db" + str(l + 1)])
    
    def model(self):
        for i in range(self.num_iterations):
            self.feed_forward()
            self.compute_cost()
            self.backpropagate()
            self.update_parameters()
            if i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(self.cost)))
                self.costs.append(self.cost)


    def predict(self, X):
        """
        Predicts the output for input data X.

        Args:
        X (numpy.ndarray): Input data of shape (input_size, num_examples).

        Returns:
        predictions (numpy.ndarray): Model predictions of shape (output_size, num_examples).
        """
        # Ensure X has the correct shape
        assert X.shape[0] == self.layers_dims[0], f"Input data must have shape ({self.layers_dims[0]}, num_examples)."

        # Perform forward propagation to get predictions
        self.train_X = X
        self.feed_forward()
        
        # Return the predictions (A[L] for the last layer)
        return self.AL


if __name__ == "__main__":
    layers_dims = [1, 20, 7, 5, 1]  # num of neurons of each layer. input-layer-1, output-layer-1. hidden-layers-20-7-5. Layer-0 is input-layer. Layer-4 is output-layer
    # X.shape = (n[0], m). Num of rows is number of input-neurons. Num of cols is number of examples. 
    train_x = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.5, 0.5, 0.1, 0.2, 0.12, 2.3, 0.49, 2.34, 1.12, 1.26, 1.47, 0.69, 0.96, 2.4, 4.5, 6.8, 8.9, 0.2, 9.9, 8.8, 6.4, 5.9, 3.9, 2.9, 9.4, 8.3, 3.6, 9.7, 4.0, 2.6, 2.5, 6.7, 6.5, 6.6, 6.2, 5.0, 2.4, 6.1, 5.3, 3.2, 3.1, 2.1, 8.1, 8.4, 8.5, 9.1, 9.6, 6.6, 1.1, 7.5, 7.1, 7.4, 8.6, 6.8, 2.7, 7.2, 9.6, 5.4, 9.7, 3.5, 2.1, 7.7, 8.8, 4.0, 3.9, 8.5, 7.0, 1.0, 3.0, 4.0, 8.0, 9.0, 9.2, 9.4, 4.4, 3.3, 1.1, 2.2, 8.8, 7.6, 6.8]
    ])
    # Y.shape = (n[L], m). Num of rows is number of output-neurons. Num of cols is number of examples
    train_y = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    ])

    nn = FeedForwardNN(train_x, train_y, layers_dims, 0.0075, 2500) # create NN-obj passing in training input-features-X and output-labels-Y
    nn.model() # run NN-model

    iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, 2500) # og on both is (0, 900)
    plt.ylim(0.68, 0.70)
    plt.show()

# Note: each layer including the input-layer and output-layer are indexed from 0 to num-layers-1. The W1 is before layer-1. The dA-last-layer 
# is after the output-layer which is dAL=predictions. The dA3 is after layer-3. The dA0-input-layer is after input-layer. dW1 is before layer-1. 
# Each group is always [dA[l-1], dW[l], db[l]]. 

# TODO: add 4 input nodes dataset and see if cost is increasing
# 0.6859304762149067
# 0.6859306338146303