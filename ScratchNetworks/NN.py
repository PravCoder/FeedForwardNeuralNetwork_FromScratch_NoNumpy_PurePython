import numpy as np      # ONLY USING NUMPY TO PRINT SHAPES OF PYTHON MATRICIES IN show_shapes METHOD
import math      # to perform mathmatical computations
import matplotlib.pyplot as plt     # used to visualize the cost
import copy     # used as 2nd check so that the data of a object is copied and not its reference

class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, dimensions, learning_rate=0.0075, num_iterations=3000, binary_classification=False, multiclass_classification=False, regression=False, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch"):
        self.X = X      # x-training data
        self.Y = Y      # y-training data
        self.saved_X = self.X
        self.saved_Y = self.Y
        self.output_nodes = len(self.Y)     # number of output_nodes in last-layer, each row in y-training data represents each respective output node
        self.m = 0      # number of examples model is to be trained on
        self.dimensions = dimensions        # a list where each element is the number of nodes in that layer index 
        self.learning_rate = learning_rate      # hyperparameter alpha that denotes the step size at which optimization algorithm moves toward the minimum of cost function
        self.num_iterations = num_iterations        # total number of iterations in which the model is to be trained, 1-iteration is (forward_prop,cost,backprop,parameter_update,repeat)
        self.cost = 0       # current cost value of current iteration of the current model, measures how well the models predictions match actual target label values
        self.costs = []     # after each iteration the cost is computed and added to this list, stores all of the costs for all iterations
        self.binary_classification = binary_classification
        self.multiclass_classification = multiclass_classification
        self.regression = regression
        # PARAMETERS/GRAIDENTS REPRESENTATION:
        self.W = []     # = [weights-layer-0,weights-layer-1,...,weights-layer-L], weights-layer-1 = [node0-layer-0, node1-layer-0,...,nodeL-layer-0] each element is a list representing each node in previous-layer list has length of number of nodes in prev-layer, node0-layer-0 = [w1,w2,..,w(n[l-1 * l])] each element is the weight-value connected to that node in previous layer,the list of length n[l-1]*n[l]. self.W[l][prev] = list of weight-values connected to the node in l-1 with index prev
        self.b = []     # = [biases-layer-0,biases-layer-1,...,biases-layer-L] each element is a list representing the biases for that layer, biases-layer-1 = [b1,b2,b3,...,b(n[l])] each element is the bias-value for that node in current layer. self.b[l] = length n[l]
        self.dW = []        # same representation/shape as self.W
        self.db = []        # same representation/shape as self.b
        # ACTIVATIONS/WEIGHTED-SUM/DERIVATIVE OF COST RESPECT TO ACTIVATIONS AND WEIGHTED-SUMS: self.Z[l][m][node]/self.A[l][m][node]
        self.Z = []     # [z-layer-0,z-layer-1,...,z-layer-L] each element is a list representing the weighted-sums for each layer, z-layer-1 = [z1(m1),z1(m2),...,z1(m)] each element is a list representing the weighted-sums for current-layer and that example, z1(m1) = = [z1,z2,z3,...,z(n[l])] each element is the weighted-sum value for current layer and that example
        self.A = []     # [a-layer-0,a-layer-1,...,a-layer-L] each element is a list representing the activations for each layer, a-layer-1 = [a1(m1),a1(m2),...,a1(m)] each element is a list representing the activations for current-layer and that example, a1(m1) = = [a1,a2,a3,...,a(n[l])] each element is the activations value for current layer and that example
        self.dZ = []        # same representation/shape as self.Z
        self.dA = []        # same representation/shape as self.As
        self.epsilon = 1e-8         # constant value used to approxmiat ethe derivative of cost function weith respect to each individual parameter
        # REGRESSION TASKS:
        self.accurate_threshold = 0.05  # determines if a prediction is within the range of a correct prediction for regression tasks
        self.accuracy_method = "MAE" # 'MAE' or 'tolerence' are 2 different ways to evaluating accuracy of a regression-model

    def get_input_info(self):
        for row in self.X:
            for _ in row:
                self.m += 1
            #print("m-examples: " + str(self.m))
            self.examples_saved = self.m
            break
    def get_num_examples(self):     # iterate or find the length of 1 row in input-matrix to total examples 
        ex = 0
        for row in self.X:
            for _ in row:
                ex += 1
            return ex
    def show_shapes(self, X, string):
        for i, layer in enumerate(X):
            print(string + "" + str(i) + ": " + str(np.array(layer).shape))
            return

    def initialize_parameters(self):          
        for _ in range(0, len(self.dimensions)):     
            self.W.append([])
            self.b.append([])
        for l in range(1, len(self.dimensions)):      
            for prev in range(self.dimensions[l-1]):  
                self.W[l].append([])
                for next in range(self.dimensions[l]):    
                    # float(np.random.randn() *0.01))
                    self.W[l][prev].append(np.random.randn() *0.01) # UNCOMMENT THIS! 0.018230189162016477
            for _ in range(self.dimensions[l]):     
                self.b[l].append(0)


    def initialize_calculations(self):   
        for _ in range(0, len(self.dimensions)):        # iterate through each layer index and add empty list to Z/A
            self.Z.append([])
            self.A.append([])
        for l in range(0, len(self.dimensions)):        # iterate through each layer index again but index of current layer is needed
           for c in range(self.dimensions[l]):
               self.Z[l].append([])
               self.A[l].append([])
               for _ in range(self.m):
                   self.Z[l][c].append(0)
                   self.A[l][c].append(0)

    def initialize_gradients(self):    
        for _ in range(0, len(self.dimensions)):      
            self.dW.append([])
            self.db.append([])
        for l in range(1, len(self.dimensions)):      
            for prev in range(self.dimensions[l-1]):        # iterate all indicies of each node in previous layer and add empty list to W
                self.dW[l].append([])
                for next in range(self.dimensions[l]):      # iterate all indicies of each node in current layer and use prev-node-index to add a initial weight value for that prev-node in current layer to W[l][prev]. Number of weights in a layer is n[l]*n[l-1] so using nested loop
                    self.dW[l][prev].append(0) # UNCOMMENT THIS!
            for _ in range(self.dimensions[l]):      # iterate each node index in current layer and add 0 for bias for current layer, there is a seperate bias for each node in each layer
                self.db[l].append(0)

        for _ in range(0, len(self.dimensions)):        # iterate through each layer index and add empty list to Z/A
            self.dZ.append([])
            self.dA.append([])
        for l in range(0, len(self.dimensions)):        # iterate through each layer index again but index of current layer is needed
           for c in range(self.dimensions[l]):
               self.dZ[l].append([])
               self.dA[l].append([])
               for _ in range(self.m):
                   self.dZ[l][c].append(0)
                   self.dA[l][c].append(0)
        #self.show_shapes(self.dW, "dW")
        #self.show_shapes(self.db, "db")
        #self.show_shapes(self.dZ, "dZ")
        #self.show_shapes(self.dA, "dA")

    def forward_propagation(self, predict=False, show_predictions=False, acutal_y=None):        # computes the weighted sum and activations for each node in network
        for m in range(self.m):
            for cur in range(self.dimensions[0]):
                self.A[0][cur][m] = self.X[cur][m]
        for l in range(1, len(self.dimensions)):
            for m in range(self.m):
                for cur in range(self.dimensions[l]):
                    self.Z[l][cur][m] = 0
                    for prev in range(self.dimensions[l - 1]):
                        self.Z[l][cur][m] += self.W[l][prev][cur] * self.A[l - 1][prev][m] 
                    self.Z[l][cur][m] += + self.b[l][cur]
                    self.A[l][cur][m] = self.relu_single(self.Z[l][cur][m])
        L = len(self.dimensions) - 1
        for m in range(self.m):
            for cur in range(self.dimensions[L]):
                for prev in range(self.dimensions[L - 1]):
                    self.Z[L][cur][m] += self.W[L][prev][cur] * self.A[L - 1][prev][m] 
                self.Z[L][cur][m] += self.b[L][cur]
                if self.binary_classification == True or self.multiclass_classification == True:
                    self.A[L][cur][m] = self.sigmoid_single(self.Z[L][cur][m])
                if self.regression == True:   # linear-activation for regression
                    self.A[L][cur][m] = self.Z[L][cur][m]


        # DECIDE PREDICTIONS
        predictions = []        # activations of the last-layer, the output values for each output node
        # BINARY-CLASSIFICATION PREDICTIONS
        if predict == True and self.binary_classification == True:      
            for n in range(self.dimensions[L]):     # iterate each output-node-index
                y_hat = self.A[L][0][n]     # activation of last-layer and 0 for 1 example because we are predicting 1 example and the index of current-output-node-n
                prediction = None      
                if y_hat >= 0.5:        # if the output value of the current node is greater than 50% prediction is 1 else 0
                    prediction = 1
                else:
                    prediction = 0
                predictions.append(prediction)      # add the current prediction of the output node to list
            if show_predictions == True:
                print("Input: " + str(self.X[0]))
                print("y-hat: " + str(y_hat))
                print("y: " + str(acutal_y))
                print("Predicted: " + str(prediction))
                print("--------")
            return predictions      # return a list of all of the predictions for binary classification
        # MULTICLASS-CLASSIFICATION PREDICTIONS
        if predict == True and self.multiclass_classification == True:    
            for n in range(self.dimensions[L]):     # iterate each output-node-index
                y_hat = self.A[L][n][0]     # activation of last-layer and 0 for 1 example because we are predicting 1 example and the index of current-output-node-n
                prediction = y_hat
                predictions.append(prediction)
            pred_i = predictions.index(max(predictions))
            if show_predictions == True:
                print("Input: " + str(self.X[0]))
                print("y-hat: " + str(y_hat)) # pre-rounded value of prediction which is max out of all output nodes for multiclass classification
                print("Predicted: " + str(prediction)) # binary rounded
                print("--------")
            return pred_i, predictions
        # REGRESSION PREDICTIONS:
        if predict == True and self.regression == True:
            for n in range(self.dimensions[L]):     # iterate each output-node-index
                y_hat = self.A[L][0][n]  
                predictions.append(y_hat)
            if show_predictions == True:
                print("Input: " + str(self.X[0]))
                print("y-hat: " + str(y_hat)) # pre-rounded value of prediction which is max out of all output nodes for multiclass classification
                print("Predicted: " + str(prediction)) # binary rounded
                print("--------")
            return predictions


    def compute_cost(self):     # BINARY-CLASSIFICATION
        L = len(self.dimensions)-1      # get index of last-layer
        AL = self.A[L]      
        total_cost = 0
        for cur in range(self.dimensions[L]):     # iterate throughe each example index
            for m in range(self.m):     # iterate each node index in last-layer
                example_cost = 0
                y = self.Y[cur][m]        # get the actual label of current ouput-node and current example
                al = AL[cur][m]           # get the activation of current example and current output-node of last-layer
                example_cost += y * np.log(al) + (1 - y) * np.log(1 - al)
                total_cost += example_cost      # average the example cost and add it to total-cost
        self.cost = -total_cost / self.m

    def compute_cost_MSE(self):  # MEAN-SQUARED-ERROR: cost function for regression
        L = len(self.dimensions)-1      
        AL = self.A[L]      
        total_cost = 0
        for n in range(self.dimensions[L]): 
            for m in range(self.m):     
                y = self.Y[n][m]        
                al = AL[n][m]           
                total_cost += math.pow(al - y, 2)
        self.cost = total_cost / self.m

    def sigmoid_single(self, x):
        return  1/(1+math.exp(-x))

    def sigmoid_backward_single(self, z):
        return self.sigmoid_single(z) * (1-self.sigmoid_single(z))
    def relu_single(self, x):  # TODO: fix relu formula? MAX or return 1/0??????
        if x >= 0:
            return 1
        else:
            return 0
    def relu_backward_single(self, z):
        if z > 0:
            return 1.0  # Derivative of ReLU when z > 0 is 1
        else:
            return 0.0  # Derivative of ReLU when z <= 0 is 0

    def backward_propagation(self):
        L = len(self.dimensions) - 1        # get index of last-layer
        # INITIALIZE GRADIENTS
        self.dW, self.db = [], []       # at each iteration initialize gradients like
        for _ in range(0, len(self.dimensions)):
            self.dW.append([])
            self.db.append([])
        for l in range(1, len(self.dimensions)):
            for prev in range(self.dimensions[l-1]):
                self.dW[l].append([])
                for next in range(self.dimensions[l]):
                    self.dW[l][prev].append(0)
            for cur in range(self.dimensions[l]):
                self.db[l].append(0)

        for m in range(self.m):
            for cur in range(self.dimensions[L]):
                y = self.Y[cur][m]
                al = self.A[L][cur][m]
                self.dA[L][cur][m] = -((y / al) - ((1 - y) / (1 - al)))
        
        # LAST LAYER:
        for cur in range(self.dimensions[L]):
            for m in range(self.m):
                self.dZ[L][cur][m] = self.dA[L][cur][m] * self.sigmoid_backward_single(self.Z[L][cur][m])
        
        for prev in range(self.dimensions[L-1]):
            for cur in range(self.dimensions[L]):
                for m in range(self.m):
                    self.dW[L][prev][cur] += self.dZ[L][cur][m] * self.A[L-1][prev][m]
                    self.db[L][cur] += self.dZ[L][cur][m]
                self.dA[L-1][prev][m] += self.W[L][prev][cur] * self.dZ[L][cur][m]

        # REST OF THE LAYERS:
        for l in range(len(self.dimensions)-2, 0, -1):
            for cur in range(self.dimensions[l]):
                for m in range(self.m):
                    self.dZ[l][cur][m] = self.dA[l][cur][m] * self.relu_backward_single(self.Z[l][cur][m])
            
            for prev in range(self.dimensions[l-1]):
                for cur in range(self.dimensions[l]):
                    for m in range(self.m):
                        self.dW[l][prev][cur] += self.dZ[l][cur][m] * self.A[l-1][prev][m]
                        self.db[l][cur] += self.dZ[l][cur][m]
                    self.dA[l-1][prev][m] += self.W[l][prev][cur] * self.dZ[l][cur][m]

        # AVERAGE GRADIENTS
        for l in range(len(self.dimensions) - 1, 0, -1):
            for cur in range(self.dimensions[l]):
                for prev_node in range(self.dimensions[l - 1]):
                    self.dW[l][prev_node][cur] /= self.m
                self.db[l][cur] /= self.m

    def update_parameters_gradient_descent(self):
        for l in range(1, len(self.dimensions)):
            for cur in range(self.dimensions[l]):
                for prev in range(self.dimensions[l-1]):
                    self.W[l][prev][cur] = self.W[l][prev][cur] - (self.learning_rate * self.dW[l][prev][cur])
                self.b[l][cur] = self.b[l][cur] - (self.learning_rate * self.b[l][cur])
                
    def predict(self, x, y=[], show_preds=False): 
        new_x = x
        new_y = y
        self.m = 0
        self.X, self.Y = new_x, new_y
        self.get_input_info()
        self.initialize_parameters()
        self.initialize_calculations()
        self.initialize_gradients()
        p = self.forward_propagation(predict=True, show_predictions=False, acutal_y=y)
        if show_preds == True:
            print("Network Predictions (max-indx, output-values): " + str(p))
        return p

    def evaluate_accuracy(self): # passes 1 example at a time through network to evaluate output
        L = len(self.dimensions) - 1        # get index of last-layer
        x_train, y_train = self.X, self.Y       # look at the structure nad representation of the data at the bottom of the file
        num_correct = 0     # stores number of examples guessed correctly by model
        num_examples = self.get_num_examples()      # get total number of examples

        for ex in range(self.m):        # iterate through each example index
            inputs = []     # stores the inputs for current example, each row represents each input node
            outputs = []        # stores the outputs for current example, each row represents each output node
            for input_node in range(self.dimensions[0]):        # iterate through each input-node-index and add the feature of the current input-node and example in a list to inputs
                inputs.append([x_train[input_node][ex]])
            for output_node in range(self.dimensions[L]):       # iterate through each output-node-index and add the label of the current output-node and example in a list to outputs
                outputs.append([y_train[output_node][ex]])
            #  BINARY CLASSIFICATION: DETERMINE CORRECT PREDICTIONS
            if self.binary_classification == True:     
                preds = self.predict(inputs, outputs)       # get the predictions given the inputs in a list, preds = [p1,p2,p3]
                is_correct = True
                for i, p in enumerate(preds):       # iterate through each prediction although we have 1 prediction, if the current prediction is not equal to the label of the current output-node, 0 because we have 1 example
                    if p != outputs[i][0]:
                        is_correct = False
                if is_correct:      # if the current prediction is correct increment number of correct guesses
                    num_correct += 1
            # MULTICLASS-CLASSIFICATION: DETERMINE CORRECT PREDICTIONS
            if self.multiclass_classification == True:      
                prediction_indx, preds = self.predict(inputs, outputs)      # get the index of the class/output-node with highest problaity and all of the predictions in list preds
                is_correct = True
                output_indx = 0     # stores the index of output-node-label with the highest probality       
                max_output_val = 0      # basically trying to find the actual greatest output label adn what output-node that value belongs to
                for i in range(self.dimensions[L]):     # itearte through each output-node-index
                    if outputs[i][0] > max_output_val:      # if the current output-node-lbale is greater thatn the max-output-value, update max_output_value and the index of correct label class node
                        max_output_val = outputs[i][0]
                        output_indx = i 
                if prediction_indx == output_indx:      # if the index of output-node of the prediction is equal to the index of the label it is a correct prediction
                    num_correct += 1
            # REGRESSION-PROBLEM: DETERMINE ACCURACY USING 2 DIFFERENT METHODS
            if self.regression == True:     
                preds = self.predict(inputs, outputs)      # length prediction = number of units in output layer
                # Accuracy within Tolerence:
                if self.accuracy_method == "tolerence":
                    for n in range(self.dimensions[L]):
                        if abs(outputs[n][0] - preds[n]) <= self.accurate_threshold:
                            num_correct += 1
                # Mean Absolute Error (MAE):
                if self.accuracy_method == "MAE":
                    total_sum = 0
                    for i in range(len(preds)):
                        total_sum += abs(outputs[i][0] - preds[i])
                    accuracy = (1/self.m) * total_sum

        # PRINT ACCURACY: REGRESSION
        if self.regression == True:
            if self.accuracy_method == "tolerence":
                percentage = num_correct/num_examples
                print("---------------")
                print("Correct: " + str(num_correct))
                print("Examples: " + str(num_examples))
                print("Accuracy Tolerence: " + str(percentage))
                print("---------------")
            if self.accuracy_method == "MAE":
                print("---------------")
                print("Accuracy: MAE: " + str(accuracy))
                print("---------------")
        # PRINT ACCURACY: BINARY/MULTI-CLASSIFICATION
        else:
            percentage = num_correct/num_examples
            print("---------------")
            print("Correct: " + str(num_correct))
            print("Examples: " + str(num_examples))
            print("Accuracy: " + str(percentage))
            print("---------------")
    def print_info(self):
        print("Neural Network Dimensions: " + str(self.dimensions))
        print("Examples m: " + str(self.m))
        if self.regression == True:
            print("Cost Function: Mean-Squared-Error " + self.accuracy_method)
        if self.regression == False:
            print("Cost Function: Cross-Entropy")

        print("-----------------------")
    def train(self):
        self.get_input_info()
        self.initialize_parameters()
        self.initialize_calculations()
        self.initialize_gradients()
        self.print_info()
        for i in range(self.num_iterations):
            self.forward_propagation()
            if self.binary_classification == True or self.multiclass_classification == True:
                self.compute_cost()
            if self.regression == True:
                self.compute_cost_MSE()
            self.backward_propagation()   # DISABLE BACKPROP TO PRINT GRADIENTS
            self.update_parameters_gradient_descent()

            if i % 100 == 0 or i == self.num_iterations - 1:
                print('Cost after {} iterations is {}'.format(i, self.cost))
                self.costs.append(self.cost)

# STATUS: Increasing cost for single/multiple output neurons possible due to incorrect backprop implementation and cost function.
layers_dims = [4, 2, 1]  # num of neurons of each layer

# each row represents inputs for each input node, each element in a row are all the example input values for that input node. To get all of the inputs for a specific example use that same index in each row
"""train_x = [
    [0.1, 0.2, 0.3],
]
# each row represents outputs for each output node, each element in a row are all the example output values for that output node. To get all of the outputs for a specific example use that same index in each row
train_y = [
    [0, 1, 0],
]"""
train_x = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.5, 0.5, 0.1, 0.2, 0.12, 2.3, 0.49, 2.34, 1.12, 1.26, 1.47, 0.69, 0.96, 2.4, 4.5, 6.8, 8.9, 0.2, 9.9, 8.8, 6.4, 5.9, 3.9, 2.9, 9.4, 8.3, 3.6, 9.7, 4.0, 2.6, 2.5, 6.7, 6.5, 6.6, 6.2, 5.0, 2.4, 6.1, 5.3, 3.2, 3.1, 2.1, 8.1, 8.4, 8.5, 9.1, 9.6, 6.6, 1.1, 7.5, 7.1, 7.4, 8.6, 6.8, 2.7, 7.2, 9.6, 5.4, 9.7, 3.5, 2.1, 7.7, 8.8, 4.0, 3.9, 8.5, 7.0, 1.0, 3.0, 4.0, 8.0, 9.0, 9.2, 9.4, 4.4, 3.3, 1.1, 2.2, 8.8, 7.6, 6.8],
    [0.09, 0.19, 0.28, 0.37, 0.51, 0.64, 0.75, 0.83, 1.05, 1.12, 1.24, 1.33, 1.44, 1.52, 1.66, 1.73, 1.82, 1.96, 2.01, 1.52, 0.52, 0.08, 0.19, 0.11, 2.27, 0.47, 2.31, 1.16, 1.29, 1.37, 0.60, 0.91, 2.26, 4.43, 6.65, 8.81, 0.18, 9.74, 8.72, 6.29, 5.86, 3.81, 2.85, 9.41, 8.23, 3.58, 9.68, 3.95, 2.51, 2.49, 6.61, 6.46, 6.60, 6.07, 4.94, 2.32, 6.03, 5.31, 3.11, 3.09, 1.97, 8.06, 8.38, 8.40, 9.04, 9.53, 6.46, 1.02, 7.47, 7.00, 7.42, 8.47, 6.72, 2.68, 7.16, 9.57, 5.47, 9.78, 3.44, 1.99, 7.70, 8.82, 3.95, 3.88, 8.40, 7.09, 0.98, 2.90, 3.95, 8.06, 8.98, 9.19, 9.48, 4.35, 3.23, 1.00, 2.15, 8.68, 7.48, 6.64],
    [0.12, 0.21, 0.35, 0.42, 0.54, 0.61, 0.78, 0.85, 1.02, 1.14, 1.28, 1.30, 1.42, 1.59, 1.68, 1.78, 1.85, 1.92, 2.06, 1.45, 0.46, 0.13, 0.26, 0.18, 2.32, 0.55, 2.39, 1.18, 1.23, 1.42, 0.67, 0.87, 2.44, 4.59, 6.70, 8.94, 0.27, 9.89, 8.77, 6.47, 5.91, 3.92, 2.88, 9.49, 8.32, 3.64, 9.78, 4.12, 2.68, 2.58, 6.72, 6.51, 6.68, 6.18, 4.96, 2.42, 6.12, 5.33, 3.24, 3.18, 2.08, 8.19, 8.45, 8.57, 9.10, 9.63, 6.73, 1.12, 7.53, 7.16, 7.54, 8.63, 6.86, 2.78, 7.26, 9.69, 5.65, 9.89, 3.65, 2.16, 7.77, 8.91, 4.18, 3.98, 8.60, 7.05, 1.10, 3.12, 4.16, 8.16, 9.03, 9.25, 9.58, 4.47, 3.38, 1.16, 2.23, 8.88, 7.68, 6.81],
    [0.15, 0.24, 0.32, 0.49, 0.50, 0.69, 0.73, 0.88, 1.01, 1.19, 1.23, 1.36, 1.40, 1.57, 1.65, 1.72, 1.87, 1.98, 2.12, 1.59, 0.59, 0.15, 0.29, 0.16, 2.25, 0.53, 2.28, 1.15, 1.32, 1.44, 0.63, 0.95, 2.45, 4.49, 6.77, 8.85, 0.23, 9.81, 8.84, 6.41, 5.83, 3.88, 2.92, 9.47, 8.36, 3.66, 9.72, 4.10, 2.61, 2.51, 6.67, 6.53, 6.67, 6.27, 4.98, 2.48, 6.17, 5.34, 3.18, 3.19, 2.15, 8.10, 8.39, 8.56, 9.18, 9.70, 6.64, 1.10, 7.54, 7.02, 7.47, 8.67, 6.80, 2.76, 7.22, 9.65, 5.56, 9.76, 3.48, 2.10, 7.74, 8.85, 4.10, 3.95, 8.53, 7.03, 0.95, 2.96, 3.92, 8.08, 8.94, 9.21, 9.45, 4.33, 3.28, 1.06, 2.28, 8.78, 7.58, 6.86]
]
# each row represents outputs for each output node, each element in a row are all the example output values for that output node. To get all of the outputs for a specific example use that same index in each row
train_y = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
]


if __name__ == "__main__":
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, binary_classification=True)
    nn.train()
    """for l in range(1, len(nn.dimensions)):
        print("LAYER: " + str(l))
        for cur in range(nn.dimensions[l]):
            for m in range(nn.m):  # Z[node][example]
                print("Z["+str(cur+1)+"]"+"["+str(m+1)+"]: "+str(nn.Z[l][cur][m]))
    print("\n")
    for l in range(1, len(nn.dimensions)):
        print("LAYER: " + str(l))
        for cur in range(nn.dimensions[l]):
            for m in range(nn.m):  # Z[node][example]
                print("A["+str(cur+1)+"]"+"["+str(m+1)+"]: "+str(nn.A[l][cur][m]))"""
    
    """iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, 2500)
    plt.ylim(0.68, 0.70)
    plt.show()"""




# NETWORK STATUS:
# ocasionally decreases consistently
# unpredictable behavoir
# most of the time is approximately constant. 

# Cost decreasing for always for constat weight initialzation.

# NOTE:
# - forward_prop calculations are matching paper calculations
# - cost decreasing to 0.63 for 3-examples.
# - majority of cost is constant for 
# - sine surve predictions are all the same
# - cost decreasing more times, sometimes constant.

# TODO:
# 1) compute on network on paper with constant weights and print network compuataions and compare
# 2) Relu activation: max yields constant cost returning 0/1 yields decreasing cost.
# 3) Constant cost with 100 examples.
