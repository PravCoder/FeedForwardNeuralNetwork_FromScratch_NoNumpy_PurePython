import numpy as np      # ONLY USING NUMPY TO PRINT SHAPES OF PYTHON MATRICIES IN show_shapes METHOD
import math as mth      # to perform mathmatical computations
import matplotlib.pyplot as plt     # used to visualize the cost
import copy     # used as 2nd check so that the data of a object is copied and not its reference

class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, dimensions, learning_rate=0.0075, num_iterations=3000, l2_regularization=False, binary_classification=False, multiclass_classification=False, regression=False, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch"):
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
        self.dA = []        # same representation/shape as self.A
        # GRADIENT CHECKING:
        self.theta_plus = []        # copy of self.W but each individual weight-value for each layer is incremented by the epsilon value
        self.theta_minus = []       # copy of self.W but each individual weight-value for each layer is decremented by the epsilon value
        self.epsilon = 1e-8         # constant value used to approxmiat ethe derivative of cost function weith respect to each individual parameter
        # L2-REGULARIZATION:
        self.lambd = 0      # hyperparameter that dictates the strength of regularization term in cost-function and gradient calculations
        self.l2_regularization = l2_regularization      # boolean value that tells if l2-reg should be implmented in training proccess
        # OPTIMIZATION METHODS:
        self.optimizer = optimizer  # gradient-descent, adam, momentum
        self.vdw = []
        self.vdb = []
        self.beta1 = 0.9
        self.sdw = []
        self.sdb = []
        self.beta2 = 0.999
        self.epoch_num = 1
        # GRADIENT DESCENT VARIATIONS:
        self.gradient_descent_variant = gradient_descent_variant  # batch, stochastic, mini-batch
        self.mini_batch_size = 1
        self.num_mini_batches = None
        self.x_mini_batches = []  # stores x-mini-batch data, each element is in the form of a small dataset or mini-batch
        self.y_mini_batches = []
        self.examples_saved = self.m  # saves the total number of examples in dataset, not in mini-batches
        # LEARNING RATE METHODS:
        self.learning_rate_decay = learning_rate_decay  # true if using scheduled learning rate decay
        self.learning_rate0 = self.learning_rate   # inital learning rate at first iteration
        self.decay_rate = 0.3
        self.time_interval = 100
        # REGRESSION TASKS:
        self.accurate_threshold = 0.05  # determines if a prediction is within the range of a correct prediction for regression tasks
        self.accuracy_method = "MAE" # 'MAE' or 'tolerence' are 2 different ways to evaluating accuracy of a regression-model
        # BATCH NORMALIZATION:
        self.z_norm = []
        self.z_tilda = []
        self.gamma = []
    
    def check_gradients(self):
        print("------")
        self.theta_plus = copy.deepcopy(self.W)
        self.theta_minus = copy.deepcopy(self.W)

        for l in range(1, len(self.dimensions)):
            for prev in range(self.dimensions[l-1]):
                for n in range(self.dimensions[l]):
                    self.theta_plus[l][prev][n] += self.epsilon
                    self.theta_minus[l][prev][n] -= self.epsilon
        print("Thetas Equal: "+str(self.theta_minus == self.theta_plus))
        J_plus = copy.deepcopy(self.W)
        J_minus = copy.deepcopy(self.W)

        graddpprox = copy.deepcopy(self.W)
        for l in range(1, len(self.dimensions)):
            for prev in range(self.dimensions[l-1]):
                for n in range(self.dimensions[l]):
                    self.W = copy.deepcopy(self.theta_plus)
                    self.forward_propagation()
                    self.compute_cost()
                    J_plus[l][prev][n] = self.cost
                    #print("+: " + str(J_plus[l][z]))
                    #print(self.W == self.theta_plus)
                    
                    self.W = copy.deepcopy(self.theta_minus)
                    self.forward_propagation()
                    self.compute_cost()
                    J_minus[l][prev][n] = self.cost
                    #print("-: " + str(J_minus[l][z]))
                    #print(self.W == self.theta_minus)

                    graddpprox[l][prev][n] = (J_plus[l][prev][n] - J_minus[l][prev][n]) / (2*self.epsilon)
                    #print("g: "+str(graddpprox[l][z]))
                    #print("-------")
        
        grad = self.gradients_to_vector(self.dW)
        graddpprox = self.gradients_to_vector(graddpprox)
        print("grads-len: " + str(len(grad) == len(graddpprox)))
        numerator = self.l2_norm(self.subtract_vector(grad, graddpprox))
        denominator = self.l2_norm(grad) + self.l2_norm(graddpprox) 
        print("num: " + str(numerator))
        print("den: " + str(denominator))

        difference = numerator/denominator

        if difference > 1e-7:
            print("Gradient Checking.....Incorrect backpropagation. Difference: " + str(difference))
        else:
            print("Gradient Checking.....Correct backpropagation! Difference: " + str(difference))

    def gradients_to_vector(self, grads):
        vector = []
        for l in range(1, len(self.dimensions)):
            for prev in range(self.dimensions[l-1]):
                for n in range(self.dimensions[l]):
                    vector.append(grads[l][prev][n])
        return vector
    def l2_norm(self, vector):
        sum_squares = sum(mth.pow(x, 2) for x in vector)
        return mth.sqrt(sum_squares)
    def subtract_vector(self, v1,v2):
        for i, val in enumerate(v1):
            v1[i] = val - v2[i]
        return v1
    def get_input_info(self):
        #print("X: "+str(self.X))
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

    def initialize_parameters(self):        # creates inital representation of the parameters           
        for _ in range(0, len(self.dimensions)):        # iterate through each layer and add empty list to W/b
            self.W.append([])
            self.b.append([])
        for l in range(1, len(self.dimensions)):        # iterate from 1st layer to last layer index
            for prev in range(self.dimensions[l-1]):        # iterate all indicies of each node in previous layer and add empty list to W
                self.W[l].append([])
                for next in range(self.dimensions[l]):      # iterate all indicies of each node in current layer and use prev-node-index to add a initial weight value for that prev-node in current layer to W[l][prev]. Number of weights in a layer is n[l]*n[l-1] so using nested loop
                    self.W[l][prev].append(float(np.random.randn() *0.01)) # UNCOMMENT THIS!
                    #self.W[l][prev].append(0)  # zero weight initlization COMMENT THIS!
            for _ in range(self.dimensions[l]):      # iterate each node index in current layer and add 0 for bias for current layer, there is a seperate bias for each node in each layer
                self.b[l].append(0)
        #self.show_shapes(self.W, "W")
        #self.show_shapes(self.b, "b")

    def initialize_calculations(self):      # creates initial representation of the activations/weighted-sums
        for _ in range(0, len(self.dimensions)):        # iterate through each layer index and add empty list to Z/A
            self.Z.append([])
            self.A.append([])
        for l in range(0, len(self.dimensions)):        # iterate through each layer index again but index of current layer is needed
            temp = [0 for _ in range(self.dimensions[l])]       # for every node in current layer add a 0 to empty temporary list
            self.Z[l] = [temp for _ in range(self.m)]       # for every example add temp in a list and set the Z/A of current layer equal to this list
            self.A[l] = [temp for _ in range(self.m)]       # self.Z[l][m][node]/self.A[l][m][node]
        #self.show_shapes(self.A, "A")
        #self.show_shapes(self.Z, "Z")

    def initialize_gradients(self):     # create inital representation of gradients of parameters as same as the parameters
        for _ in range(0, len(self.dimensions)):
            self.dW.append([])
            self.db.append([])
        for l in range(1, len(self.dimensions)):
            for prev in range(self.dimensions[l-1]):
                self.dW[l].append([])
                for next in range(self.dimensions[l]):
                    self.dW[l][prev].append(float(np.random.randn() *0.01))
            for next in range(self.dimensions[l]):
                self.db[l].append(0)
        # creates initial representation of the gradient of cost function with respect to activations and weighted sum as same as activations and weighted same
        for _ in range(0, len(self.dimensions)):
            self.dZ.append([])
            self.dA.append([])
        for l in range(0, len(self.dimensions)):
            temp = [0 for _ in range(self.dimensions[l])]
            self.dZ[l] = [temp for _ in range(self.m)]
            self.dA[l] = [temp for _ in range(self.m)]
        #self.show_shapes(self.dW, "dW")
        #self.show_shapes(self.db, "db")
        #self.show_shapes(self.dZ, "dZ")
        #self.show_shapes(self.dA, "dA")

    def initialize_velocity(self):
        for _ in range(0, len(self.dimensions)):        # iterate through each layer and add empty list to W/b
            self.vdw.append([])
            self.vdb.append([])
        for l in range(1, len(self.dimensions)):        # iterate from 1st layer to last layer index
            for prev in range(self.dimensions[l-1]):        # iterate all indicies of each node in previous layer and add empty list to W
                self.vdw[l].append([])
                for next in range(self.dimensions[l]):      # iterate all indicies of each node in current layer and use prev-node-index to add a initial weight value for that prev-node in current layer to W[l][prev]. Number of weights in a layer is n[l]*n[l-1] so using nested loop
                    self.vdw[l][prev].append(0)
            for _ in range(self.dimensions[l]):      # iterate each node index in current layer and add 0 for bias for current layer, there is a seperate bias for each node in each layer
                self.vdb[l].append(0)
        #self.show_shapes(self.W, "W")
        #self.show_shapes(self.b, "b")

    def initialize_adam(self):
        for _ in range(0, len(self.dimensions)):        # iterate through each layer and add empty list to W/b
            self.vdw.append([])
            self.vdb.append([])
            self.sdw.append([])
            self.sdb.append([])
        for l in range(1, len(self.dimensions)):        # iterate from 1st layer to last layer index
            for prev in range(self.dimensions[l-1]):        # iterate all indicies of each node in previous layer and add empty list to W
                self.vdw[l].append([])
                self.sdw[l].append([])
                for next in range(self.dimensions[l]):      # iterate all indicies of each node in current layer and use prev-node-index to add a initial weight value for that prev-node in current layer to W[l][prev]. Number of weights in a layer is n[l]*n[l-1] so using nested loop
                    self.vdw[l][prev].append(0)
                    self.sdw[l][prev].append(0)
            for _ in range(self.dimensions[l]):      # iterate each node index in current layer and add 0 for bias for current layer, there is a seperate bias for each node in each layer
                self.vdb[l].append(0)
                self.sdb[l].append(0)
        #self.show_shapes(self.W, "W")
        #self.show_shapes(self.b, "b")
    
    def dot_product_w_a(self, w, a, num_acts, num_nodes, layer_indx):
        z = []
        w_indx = 0
        a_indx = 0
        for n in range(num_nodes):
            a_indx = 0
            max_diff = 0
            cur_sum = 0
            if w_indx + num_acts > len(w):
                max_diff = len(w)
            else:
                max_diff = w_indx + num_acts
            for i in range(w_indx, max_diff):
                cur_sum += w[i]*a[a_indx]

                a_indx += 1
                w_indx += 1
            cur_sum += self.b[layer_indx][n]
            z.append(cur_sum)
            
        return z
    def dot_prod(self, w, a, num_acts, num_nodes, layer_indx): # No bias for each sum
        z = []
        w_indx = 0
        a_indx = 0
        for n in range(num_nodes):
            a_indx = 0
            max_diff = 0
            cur_sum = 0
            if w_indx + num_acts > len(w):
                max_diff = len(w)
            else:
                max_diff = w_indx + num_acts
            for i in range(w_indx, max_diff):
                cur_sum += w[i]*a[a_indx]

                a_indx += 1
                w_indx += 1
            z.append(cur_sum)
            
        return z

    def sigmoid(self, z):       # given z-values for certain layer in 1D-list computes the sigmoid formula on each value and returns activated list
        for i, val in enumerate(z):
            z[i] = 1/(1+np.exp(-val))
        return z
    def relu(self, z):      # given z-values for certain layer in 1D-list computes the relu formula on each value and returns activated list
        for i, val in enumerate(z):
            if val > 0:
                z[i] = val
            else:
                z[i] = 0
        return z
    def get_weights_for_layer(self, w):     # takes in self.W[l] and adds all of the weights for that layer into a 1D-list for easy dot-product computation
        weights = []
        for prev_arr in w:      # iterate through each arr that represents each node in previous-layer
            for connection in prev_arr:     # iterate through each weight that is connected to the current node in previous-layer adn add it to 1D-list
                weights.append(connection)
        return weights

    def forward_propagation(self, predict=False, show_predictions=False, acutal_y=None):        # computes the weighted sum and activations for each node in network
        for m in range(self.m):     # iterate through each example index m
            self.A[0][m] = []       # set activations to first-input-layer of current example to empty list
            for n in range(self.dimensions[0]):     # iterate each node-index in first-layer and add the input-value of current node/example-index to the activations of first-layer of current example
                self.A[0][m].append(self.X[n][m])       # the activations of the first-layer are the input values

        for l in range(1, len(self.dimensions)-1):      # iterate through each layer-index from 2nd-layer to 2nd-to-layer-layer
            for m in range(0, self.m):      # iterate through each example index
                # weighted sum of current-layer-and-example is equal to the dot product of weights in current-layer and activations of previous-layer-and-current-example, get_weights_for_layer() returns all of the weights in current-layer in 1D-list, self.A[l-1][m] = [a1,a2,a3],  also passing in additional information number of nodes in previous/current layer and the current layer-index
                self.Z[l][m] = self.dot_product_w_a(self.get_weights_for_layer(self.W[l]), self.A[l-1][m], self.dimensions[l-1], self.dimensions[l], l)     
                self.A[l][m] = self.relu(self.Z[l][m])      # activations of current-layer-and-example is equal relu of the weighted-sum of current-layer-and-example
        
        L = len(self.dimensions)-1      # get index of last-layer
        for m in range(0, self.m):      # iterate through each example-index, compute the weighted-sum and activations of last-layer but using sigmoid
            self.Z[L][m] = self.dot_product_w_a(self.get_weights_for_layer(self.W[L]), self.A[L-1][m], self.dimensions[L-1], self.dimensions[L], L)   
            self.A[L][m] = self.sigmoid(self.Z[L][m])
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
                y_hat = self.A[L][0][n]     # activation of last-layer and 0 for 1 example because we are predicting 1 example and the index of current-output-node-n
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


    def compute_cost(self):             # CROSS-ENTROPY
        L = len(self.dimensions)-1      # get index of last-layer
        AL = self.A[L]      
        total_cost = 0              
        for n in range(self.dimensions[L]):     # iterate each node index in last-layer
            example_cost = 0
            for m in range(self.m):     # iterate through each example index
                y = self.Y[n][m]        # get the actual label of current ouput-node and current example
                al = AL[m][n]           # get the activation of current example and current output-node of last-layer
                if self.l2_regularization == True:
                    reg_term = self.lambd/2 * self.square_sum(self.W[L], L)
                    example_cost += y * mth.log(al) + (1 - y) * mth.log(1 - al) +reg_term
                else:       # use activation of last-layer and actual label to compute the cost of current example
                    example_cost += y * mth.log(al) + (1 - y) * mth.log(1 - al)
            total_cost += example_cost / self.m     # average the example cost and add it to total-cost
        self.cost = -total_cost / (self.dimensions[L])

    def compute_cost_categorical(self):
        L = len(self.dimensions) - 1
        AL = self.A[L]
        total_cost = 0

        for n in range(self.dimensions[L]):
            example_cost = 0
            for m in range(self.m):
                y = self.Y[n][m]
                al = AL[m][n]

                if self.l2_regularization:
                    reg_term = self.lambd / 2 * self.square_sum(self.W[L], L)
                    example_cost += -y * mth.log(al) + reg_term
                else:
                    example_cost += -y * mth.log(al)

            total_cost += example_cost / self.m

        self.cost = total_cost / self.m  # Average over all examples

    def compute_cost_MSE(self):  # MEAN-SQUARED-ERROR: cost function for regression
        L = len(self.dimensions)-1      
        AL = self.A[L]      
        total_cost = 0
        for n in range(self.dimensions[L]): 
            for m in range(self.m):     
                y = self.Y[n][m]        
                al = AL[m][n]           
                total_cost += mth.pow(al - y, 2)
        self.cost = total_cost / self.m

    def compute_cost_mini_batch(self):       # CROSS-ENTROPY
        L = len(self.dimensions) - 1  
        AL = self.A[L]
        mini_batch_cost = 0
        for n in range(self.dimensions[L]):  
            example_cost = 0
            for b in range(self.num_mini_batches):
                y_mini = self.y_mini_batches[b]
                for m in range(self.mini_batch_size):
                    y = y_mini[n][m]
                    al = AL[m][n]
                    
                    example_cost += y * mth.log(al) + (1 - y) * mth.log(1 - al)
            mini_batch_cost = example_cost / self.mini_batch_size
        self.cost = -mini_batch_cost / self.num_mini_batches


    def square_sum(self, w, layer_num):
        vals = []
        for n in range(self.dimensions[layer_num]):
            for prev_node in range(self.dimensions[layer_num-1]):
                vals.append(w[prev_node][n])
        return sum(vals)
    
    def reg_term(self, x, prod_term, layer_num):        # w = self.W[l], multiplies prod_term to every weight in self.W[l]
        w = x
        for n in range(self.dimensions[layer_num]):     # iterate through eahc node in current-layer
            for prev_node in range(self.dimensions[layer_num-1]):       # iterate through each node index in previous-layer adn multiply prod_term to current weight
                w[prev_node][n] *= prod_term            
        return w

    def sigmoid_single(self, x):
        return  1/(1+mth.exp(-x))
    def sigmoid_backward(self, z):
        for i, val in enumerate(z):
            z[i] = self.sigmoid_single(val) * (1-self.sigmoid_single(val))
        return z
    def relu_single(self, x): # TODO: which relu version activation to use?
        """if x > 0:
            return 1
        else:
            return 0"""
        return max(0, x)
    def relu_backward(self, z):
        for i, val in enumerate(z):
            if z[i] > 0:
                z[i] = 1
            else:
                z[i] = 0
        return z


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
                    self.dW[l][prev].append(float(np.random.randn() *0.01))
            for next in range(self.dimensions[l]):
                self.db[l].append(0)

        # TODO: iterate all output-nodes, onyl doing 1 output-node
        for m in range(self.m):     # iterate each example index
            y = self.Y[0][m]        # get actual label of current example of 0th output-node
            al_arr = self.A[L][m]       # get activations of last-layer of current layer in list
            self.dA[L][m] = []      # init derivative of activations of last-layer of current example to empty list
            for i, val in enumerate(al_arr):        # iterate each activation value in last-layer of current example
                al = val
                self.dA[L][m].append(-((y / al) - ((1 - y) / (1 - al))))        # compute dA of last-layer of current example using acutal label of current example and activation of last-layer
                
        # LAST LAYER:
        for m in range(self.m):     # iterate each example index
            for n in range(self.dimensions[L]):     # iterate node index of last-layer
                self.dZ[L][m][n] = self.dA[L][m][n] * self.sigmoid_backward(self.Z[L][m])[n]        # compute dZ-last-layer-current-example-node = dA-last-layer-current-example-node * sigmoid_derivative(Z-last-layer-current-example)-node
                for prev_node in range(self.dimensions[L - 1]):     # iterate each index of nodes in previous-layer
                    if self.l2_regularization == True:      # adding regularization term for l2-reg gradietn computation   
                        self.dW[L][prev_node][n] += self.dZ[L][m][n] * self.A[L - 1][m][prev_node] + self.reg_term(self.W[L], self.lambd/self.m, L)[prev_node][n]
                    else:       # normal no l2-reg
                        self.dW[L][prev_node][n] += self.dZ[L][m][n] * self.A[L - 1][m][prev_node]
                self.db[L][n] += self.dZ[L][m][n]
            for n2 in range(self.dimensions[L - 1]):
                if self.dimensions[L - 1] > self.dimensions[L]:    
                    self.dA[L - 1][m][n2] = self.dot_prod(self.get_weights_for_layer(self.W[L]), self.dZ[L][m], self.dimensions[L], self.dimensions[L - 1], L)[n2]
                if self.dimensions[L - 1] <= self.dimensions[L]:
                    self.dA[L - 1][m][n2] = self.dot_prod(self.get_weights_for_layer(self.W[L]), self.dZ[L][m], self.dimensions[L - 1], self.dimensions[L], L)[n2]

        # REST OF THE LAYERS:
        for l in range(len(self.dimensions) - 2, 0, -1):
            for m in range(self.m):
                for n in range(self.dimensions[l]):
                    self.dZ[l][m][n] = self.dA[l][m][n] * self.relu_backward(self.Z[l][m])[n]
                    for prev_node in range(self.dimensions[l - 1]):
                        if self.l2_regularization == True:
                            self.dW[l][prev_node][n] += self.dZ[l][m][n] * self.A[l - 1][m][prev_node] + self.reg_term(self.W[l], self.lambd/self.m, l)[prev_node][n]
                        else:
                            self.dW[l][prev_node][n] += self.dZ[l][m][n] * self.A[l - 1][m][prev_node]
                    self.db[l][n] += self.dZ[l][m][n]
                for n2 in range(self.dimensions[l - 1]):
                    if self.dimensions[l - 1] > self.dimensions[l]:     # if previous-layer is larger than current-layer
                        self.dA[l - 1][m][n2] = self.dot_prod(self.get_weights_for_layer(self.W[l]), self.dZ[l][m], self.dimensions[l], self.dimensions[l - 1], l)[n2]
                    if self.dimensions[l - 1] <= self.dimensions[l]:
                        self.dA[l - 1][m][n2] = self.dot_prod(self.get_weights_for_layer(self.W[l]), self.dZ[l][m], self.dimensions[l - 1], self.dimensions[l], l)[n2]

        # After the nested loops, divide the accumulated gradients by the number of examples (self.m)
        for l in range(len(self.dimensions) - 1, 0, -1):
            for n in range(self.dimensions[l]):
                # Divide each element in dW and db by the number of examples (self.m)
                for prev_node in range(self.dimensions[l - 1]):
                    self.dW[l][prev_node][n] /= self.m
                self.db[l][n] /= self.m

    def update_parameters_gradient_descent(self):
        for l in range(1, len(self.dimensions)):
            for n in range(self.dimensions[l]):
                for prev_node in range(self.dimensions[l-1]):
                    if self.l2_regularization == True:      
                        self.W[l][prev_node][n] = self.W[l][prev_node][n] - self.reg_term(self.W[l],(self.learning_rate*self.lambd)/self.m, l)[prev_node][n] - self.reg_term(self.dW[l],self.learning_rate, l)[prev_node][n]
                    else:
                        self.W[l][prev_node][n] = self.W[l][prev_node][n]  - (self.learning_rate*self.dW[l][prev_node][n])
                self.b[l][n]  = self.b[l][n]  - (self.learning_rate*self.db[l][n])

    def update_parameters_with_momentum(self):
        for l in range(1, len(self.dimensions)):
            for n in range(self.dimensions[l]):
                for prev_node in range(self.dimensions[l-1]):
                    # compute velocties for weights
                    self.vdw[l][prev_node][n] = self.beta1*self.vdw[l][prev_node][n] + (1-self.beta1)*self.dW[l][prev_node][n]
                    self.W[l][prev_node][n] = self.W[l][prev_node][n] - self.learning_rate*self.vdw[l][prev_node][n]
                # compute velocities for bias
                self.vdb[l][n] = self.beta1*self.vdb[l][n] + (1-self.beta1)*self.db[l][n]
                self.b[l][n] = self.b[l][n] - self.learning_rate*self.vdb[l][n]

    def update_parameters_with_adam(self):
        for l in range(1, len(self.dimensions)):
            for n in range(self.dimensions[l]):
                for prev_node in range(self.dimensions[l-1]):
                    # compute momentum/square vector for vdw/vdb
                    self.vdw[l][prev_node][n] = self.beta1*self.vdw[l][prev_node][n] + (1-self.beta1)*self.dW[l][prev_node][n]
                    self.sdw[l][prev_node][n] = self.beta2*self.sdw[l][prev_node][n] + (1-self.beta2)*(mth.pow(self.dW[l][prev_node][n], 2))
                    # bias correction for vdw/sdw
                    self.vdw[l][prev_node][n] = self.vdw[l][prev_node][n]/(1-mth.pow(self.beta1,self.epoch_num))
                    self.sdw[l][prev_node][n] = self.sdw[l][prev_node][n]/(1-mth.pow(self.beta2,self.epoch_num))
                    # update weights
                    self.W[l][prev_node][n] = self.W[l][prev_node][n] - self.learning_rate*(self.vdw[l][prev_node][n]/mth.sqrt(self.sdw[l][prev_node][n])+self.epsilon)
                # compute momentum/square vector for vdb/sdb
                self.vdb[l][n] = self.beta1*self.vdb[l][n] + (1-self.beta1)*self.db[l][n]
                self.sdb[l][n] = self.beta2*self.sdb[l][n] + (1-self.beta2)*(mth.pow(self.db[l][n], 2))
                # bias correction for vdb/sdb
                self.vdb[l][n] = self.vdb[l][n]/(1-mth.pow(self.beta1, self.epoch_num))
                self.sdb[l][n] = self.sdb[l][n]/(1-mth.pow(self.beta2, self.epoch_num))
                # update bias
                self.b[l][n] = self.b[l][n] - self.learning_rate*self.vdb[l][n]/(mth.sqrt(self.sdb[l][n])+self.epsilon)

    def update_learning_rate(self):
        #self.learning_rate = 1/(1+self.decay_rate*self.epoch_num) * self.learning_rate0  # decay update at each iteration
        self.learning_rate = self.learning_rate0 / (1+self.decay_rate * mth.floor(self.epoch_num/self.time_interval)) # schedule learing_rate in time intervals
                
    def predict(self, x, y, show_preds=False): 
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


    def create_mini_batches(self):
        self.num_mini_batches = self.examples_saved // self.mini_batch_size
        #print(self.examples_saved)
        x_train, y_train = self.saved_X, self.saved_Y
        for i in range(self.num_mini_batches):
            start_indx = i * self.mini_batch_size
            end_indx = (i+1) * self.mini_batch_size
            x_mini, y_mini = [], []
            
            for input_node in range(self.dimensions[0]):
                x_mini.append(x_train[input_node][start_indx: end_indx])
                
            for output_node in range(self.dimensions[-1]):
                y_mini.append(y_train[output_node][start_indx: end_indx])

            self.x_mini_batches.append(x_mini)
            self.y_mini_batches.append(y_mini)
        #print(self.x_mini_batches[1])

    def print_info(self):
        print("Neural Network Dimensions: " + str(self.dimensions))
        print("Examples m: " + str(self.m))
        if self.regression == True:
            print("Cost Function: Mean-Squared-Error " + self.accuracy_method)
        if self.regression == False:
            print("Cost Function: Cross-Entropy")
        print("Gradient Descent Varient: " + self.gradient_descent_variant)

        algorithms = [self.optimizer]
        if self.l2_regularization == True:
            algorithms.append("L2-Regularization")
        if self.learning_rate == True:
            algorithms.append("Learning Rate Decay")
        print("Algorithms: " + ", ".join(algorithms))
        print("-----------------------")


 
    def train(self):
        self.get_input_info()
        self.initialize_parameters()
        self.initialize_calculations()
        self.initialize_gradients()
        if self.optimizer == "momentum": self.initialize_velocity()
        if self.optimizer == "adam": self.initialize_adam()

        self.print_info()
        for i in range(self.num_iterations):
            # BATCH GRADIENT DESCENT
            if self.gradient_descent_variant == "batch":
                self.mini_batch_size = self.m  # change batch size of total number of examples even though we do not use it
                self.forward_propagation()
                if self.regression == True: 
                    self.compute_cost_MSE()
                if self.binary_classification == True:   # classification tasks use same cost function
                    self.compute_cost()
                if self.multiclass_classification == True:  # Can also call normal cross entropy 
                    self.compute_cost()
                self.backward_propagation()
                # check optimization algorithm
                if self.optimizer == "gradient descent":
                    self.update_parameters_gradient_descent()
                if self.optimizer == "momentum":
                    self.update_parameters_with_momentum()
                if self.optimizer == "adam":
                    self.update_parameters_with_adam()
                if self.learning_rate_decay == True:
                    self.update_learning_rate()
                #print("Weights: " + str(self.W))
                #print("Grads: " + str(self.dW))
            # MINI-BATCH GRADIENT DESCENT
            if self.gradient_descent_variant == "mini-batch":
                self.mini_batch_size = self.mini_batch_size  # redundacy batch size should already be set in _init__
                self.m = self.mini_batch_size
                self.create_mini_batches()
                for b in range(self.num_mini_batches):
                    self.X = self.x_mini_batches[b]
                    self.Y = self.y_mini_batches[b]
                    self.forward_propagation()
                    self.compute_cost_mini_batch()
                    self.backward_propagation()
                    # check optimization algorithm
                    if self.optimizer == "gradient descent":
                        self.update_parameters_gradient_descent()
                    if self.optimizer == "momentum":
                        self.update_parameters_with_momentum()
                    if self.optimizer == "adam":
                        self.update_parameters_with_adam()
                    if self.learning_rate_decay == True:
                        self.update_learning_rate()
            # STOCHASTIC GRADIENT DESCENT
            if self.gradient_descent_variant == "stochastic":
                self.mini_batch_size = 1        # change batch size to 1
                self.m = self.mini_batch_size
                self.create_mini_batches()
                for b in range(self.num_mini_batches):
                    self.X = self.x_mini_batches[b]
                    self.Y = self.y_mini_batches[b]
                    self.forward_propagation()
                    self.compute_cost_mini_batch()
                    self.backward_propagation()
                    # check optimization algorithm
                    if self.optimizer == "gradient descent":
                        self.update_parameters_gradient_descent()
                    if self.optimizer == "momentum":
                        self.update_parameters_with_momentum()
                    if self.optimizer == "adam":
                        self.update_parameters_with_adam()
                    if self.learning_rate_decay == True:
                        self.update_learning_rate()

            if i % 100 == 0 or i == self.num_iterations - 1:
                print('Cost after {} iterations is {}'.format(i, self.cost))
                self.costs.append(self.cost)
            self.epoch_num += 1

# STATUS: Increasing cost for single/multiple output neurons possible due to incorrect backprop implementation and cost function.
layers_dims = [4, 2, 3, 4, 1]  # num of neurons of each layer

# each row represents inputs for each input node, each element in a row are all the example input values for that input node. To get all of the inputs for a specific example use that same index in each row
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
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, l2_regularization=False, binary_classification=True, multiclass_classification=False, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch")
    nn.train()
    #nn.check_gradients()
    #nn.evaluate_accuracy()
    #nn.predict([[0.1],[0.09],[0.12],[0.15]], [[0], [0]], show_preds=True)

    iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, 2500)
    plt.ylim(0.68, 0.70)
    plt.show()




# TODO: Implement batch normalization.

# TODO: backpropgatiaon equations and gradient computation for MSE

# TODO: Softmax layer and cost function for multiclass-classification.

# TODO: fix math domain error for deep computations

# TODO: implement Categorical cross-entropy for multiclass classifiction cost computation