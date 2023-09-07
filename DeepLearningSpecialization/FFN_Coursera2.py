import numpy as np
import matplotlib.pyplot as plt

class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, dimensions, learning_rate=1.2, num_iterations=2500, multiclass_classification=False, regression=False):
        self.X = X
        self.Y = Y
        self.dimensions = dimensions # each element is number of nodes in that layer. Length is total number of layers. 
        self.learning_rate = learning_rate  
        self.num_iterations = num_iterations    # Number of iterations
        self.params = {}  # stores W/b parameters for each layer. {W1:np.arr, b1:np.arr}
        self.cache = {}   # stores Z/A sums for each layer. {Z1:np.arr, A1:np.arr}
        self.grads = {}   # stores dA,dZ,dW,db gradients for each layer. {dA1:np.arr, dZ1:np.arr, dW1:np.arr, db1:np.arr}
        self.cost = 0   
        self.costs = []
        self.multiclass_classification = multiclass_classification
        self.regression = regression

    def initialize(self):
        np.random.seed(3)
        # If total-layers is 5, then layer 0 and 4 are input and output layers
        # So we only need to initialize w1, w2, w3, w4, There is no need of w0 for input layer
        for l in range(1, len(self.dimensions)):
            # W[l] = np.arr.shape of (n[l], n[l-1])
            self.params["W"+str(l)] = np.random.randn(self.dimensions[l], self.dimensions[l-1])*0.01
            # b[l] = np.arr.shape of (n[l], 1])
            self.params["b"+str(l)] = np.zeros((self.dimensions[l], 1))


    def sigmoid(self, Z):     # given Z of shape(n[l], m)
        A = 1/(1+np.exp(-Z)) # activation-layer-l = sigmoid(Z) of shape(n[l], m)
        return A     
    def relu(self, Z):    # given Z of shape(n[l], m)
        A = np.maximum(0, Z) # activation-layer-l = max(0, Z) of shape(n[l], m)
        return A

    def sigmoid_backward(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def relu_backward(self, x):
        if x.all() > 0:
            return 1
        else:
            return 0

    def forward_propagation(self, predict=False):
        self.cache["A0"] = self.X  # activations for input-layer-0 = training-inputs-X
        # loop from layer-indx 1->last-hidden-layer. 1->3 for 5-total-layers
        for l in range(1, len(self.dimensions)-1):
            # Z[l] = W[l] * A[l-1] + b[l]
            self.cache["Z"+str(l)] = np.dot(self.params["W"+str(l)], self.cache["A"+str(l-1)]) + self.params["b"+str(l)]
            # A[l] = relu(Z[l])
            self.cache["A"+str(l)] = self.relu(self.cache["Z"+str(l)])
        # get index of last-output-layer. 4-indx for 5-total-layers
        l = len(self.dimensions)-1
        # Z[L] = W[L] * A[L-1] + b[L], L = index of last-output-layer
        self.cache["Z"+str(l)] = np.dot(self.params["W"+str(l)], self.cache["A"+str(l-1)]) + self.params["b"+str(l)]
        self.cache["A"+str(l)] = self.sigmoid(self.cache["Z"+str(l)])    # A[L] = sigmoid(Z[L])


    def compute_cost(self):
        m = self.Y.shape[1] # number of examples is equal to the number of columns in train-examples-Y
        L = len(self.dimensions) - 1
        AL = self.cache["A" + str(L)]

        # Calculate the categorical cross-entropy loss
        self.cost = -1/m * np.sum(np.multiply(self.Y, np.log(AL)))

        # Squeeze to convert the cost to a scalar value
        self.cost = np.squeeze(self.cost)

    def compute_cost_regression(self):
        m = self.Y.shape[1] # number of examples is equal to the number of columns in train-examples-Y
        L = len(self.dimensions) - 1
        AL = self.cache["A" + str(L)]

        self.cost = np.sum(np.power(AL - self.Y, 2)) / (2 * m)
        
    def backward_propagation(self, predict=False):
        AL = self.cache["A" + str(len(self.dimensions) - 1)] # get activation of last-layer
        m = self.X.shape[1]
        self.grads["dA"+str(len(self.dimensions)-1)] = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL)) # derivative of activation of last-layer

        L = len(self.dimensions)-1 # get index of final-layer = number of layer minus 1 because first-layer index is 0
        # Sigmoid-derivative for final layer. Set gradients of last-layer. 
        self.grads["dZ"+str(L)] = self.grads["dA"+str(L)] * self.sigmoid_backward(self.cache["Z"+str(L)])
        self.grads["dW"+str(L)] = 1 / m * np.dot(self.grads["dZ"+str(L)], self.cache["A"+str(L-1)].T)
        #print("dZ: "+ str(self.grads["dZ"+str(L)].shape))
        #print("A :" + str(self.cache["A"+str(L-1)].shape))
        self.grads["db"+str(L)] = 1 / m * np.sum(self.grads["dZ"+str(L)], axis=1, keepdims=True)
        self.grads["dA"+str(L-1)] = np.dot(self.params["W"+str(L)].T, self.grads["dZ"+str(L)]) # L-1 for activation-derivative because setting activation for last-hidden-layer. Already set dA for 
        # Loop from last-hidden-indx to input-layer-0. Relu-derivative for previous layers
        for l in range(len(self.dimensions)-2, 0, -1):
            self.grads["dZ"+str(l)] = self.grads["dA"+str(l)] * self.relu_backward(self.cache["Z" + str(l)])
            # dW[l] = dZ[l]*A[l-1].T
            self.grads["dW"+str(l)] = 1/m*np.dot(self.grads["dZ"+str(l)], self.cache["A"+str(l-1)].T)
            # db[l] = sum(dZ[l])
            self.grads["db"+str(l)] = 1/m*np.sum(self.grads["dZ"+str(l)], axis=1, keepdims=True)
            # dA[l-1] = W[l].T * dZ[l]
            self.grads["dA"+str(l-1)] = np.dot(self.params["W"+str(l)].T, self.grads["dZ"+str(l)])
    

    def update_parameters(self):
        # loop from 1st-hidden-layer to last-hidden-layer
        for l in range(1, len(self.dimensions)):
            # W[l] = W[l] - alpha*dW[l]
            self.params["W"+str(l)] = self.params["W"+str(l)] - self.learning_rate*self.grads["dW"+str(l)]
            self.params["b"+str(l)] = self.params["b"+str(l)] - self.learning_rate*self.grads["db"+str(l)]

    def train(self):
        np.random.seed(1)
        self.initialize()
        for i in range(self.num_iterations):
            #print(self.params)
            self.forward_propagation()
            if self.multiclass_classification == True:
                self.compute_cost()
            if self.regression == True:
                self.compute_cost_regression()
            self.backward_propagation()
            self.update_parameters()
            if i % 100 == 0 or i == self.num_iterations - 1:
                print('Cost after {} iterations is {}'.format(i, self.cost))
                self.costs.append(self.cost)

    def predict(self, X, outputs):
        self.cache["A0"] = X
        L = len(self.dimensions) - 1
        for l in range(1, L):
            self.cache["Z"+str(l)] = np.dot(self.params["W"+str(l)], self.cache["A"+str(l-1)]) + self.params["b"+str(l)]
            self.cache["A"+str(l)] = self.relu(self.cache["Z"+str(l)])

        self.cache["Z"+str(L)] = np.dot(self.params["W"+str(L)], self.cache["A"+str(L-1)]) + self.params["b"+str(L)]
        self.cache["A"+str(L)] = self.sigmoid(self.cache["Z"+str(L)])
        # Classify prediction for Multiclassclassification
        #print(self.cache["A"+str(L)])
        # MULTICLASS-CLASSIFICATION
        if self.multiclass_classification == True:
            preds = []
            for n in range(self.dimensions[-1]):
                preds.append(self.cache["A"+str(L)][n][0])
            return preds.index(max(preds)), preds
        # REGRESSION CLASSIFICATION
        if self.regression == True:
            preds = []
            for n in range(self.dimensions[-1]):
                preds.append(self.cache["A"+str(L)][n][0])
            return preds

    def accuracy(self, X, Y):
        L = len(self.dimensions) - 1
        m = Y.shape[1]
        accuracy = 0
        if self.multiclass_classification == True:
            predictions = self.predict(X)
            correct_predictions = np.sum(predictions == Y)
            accuracy = correct_predictions / m
        if self.regression == True:
            predictions = self.cache["A" + str(L)]
            mae = (1/m) * np.sum(np.abs(Y - predictions))
            accuracy = mae
            print("Accuracy Mean Absoulate Error: " + str(accuracy))
        return accuracy


if __name__ == "__main__":

    layers_dims = [1, 2, 3, 4, 2]  # num of neurons of each layer. input-layer-1, output-layer-1. hidden-layers-20-7-5. Layer-0 is input-layer. Layer-4 is output-layer
    # X.shape = (n[0], m). Num of rows is number of input-neurons. Num of cols is number of examples. 
    train_x = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.5, 0.5, 0.1, 0.2, 0.12, 2.3, 0.49, 2.34, 1.12, 1.26, 1.47, 0.69, 0.96, 2.4, 4.5, 6.8, 8.9, 0.2, 9.9, 8.8, 6.4, 5.9, 3.9, 2.9, 9.4, 8.3, 3.6, 9.7, 4.0, 2.6, 2.5, 6.7, 6.5, 6.6, 6.2, 5.0, 2.4, 6.1, 5.3, 3.2, 3.1, 2.1, 8.1, 8.4, 8.5, 9.1, 9.6, 6.6, 1.1, 7.5, 7.1, 7.4, 8.6, 6.8, 2.7, 7.2, 9.6, 5.4, 9.7, 3.5, 2.1, 7.7, 8.8, 4.0, 3.9, 8.5, 7.0, 1.0, 3.0, 4.0, 8.0, 9.0, 9.2, 9.4, 4.4, 3.3, 1.1, 2.2, 8.8, 7.6, 6.8]
    ])
    # Y.shape = (n[L], m). Num of rows is number of output-neurons. Num of cols is number of examples
    train_y = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    ])
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, multiclass_classification=True)
    nn.train()
    #predictions = nn.predict(train_x, [])
    #print("Predictions:", predictions)

    #accuracy = nn.accuracy(train_x, train_y)
    #print("Accuracy:", accuracy)

"""    iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)

    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, 2500) # og on both is (0, 900)
    plt.ylim(0.68, 0.70)
    plt.show()"""

    # Note: a group of relevant matricies would be W1,b1,Z1,A0