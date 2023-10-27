import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import random

# MANUAL BACKPROPAGATION EXAMPLE
class Value:
    # Value(data-number-value, [previous,value,objects], operation-type)
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data    # data of Value-obj
        self._prev = set(_children)   # stores unquie children-node previous-value-objs from children-tuple
        self._op = _op
        self._backward = lambda: None # init to empty function
        self.label = label            # string-name-label for graph
        self.grad = 0   # derivative of L respect to self-node

    # ADDITION
    def __add__(self, other):  # underscore methods override primitive operations +,-,*,/ when doing these computations with Value-objs
      other = other if isinstance(other, Value) else Value(other) # if other is primative integer convert it to value-obj
      out = Value(self.data + other.data, (self, other), "+") # output is new Value-obj with addition of data and passing in self-Value-obj and other-Value-obj as children of output

      def _backward():  # creating chain-rule backward funciton for this value for backprop
        self.grad += 1.0 * out.grad  # [local-grad * global-grad] chain-rule, global-grad = derivative of final output of expression respect to out
        other.grad += 1.0 * out.grad # output-grad is basically being copied into self-grad dand other-grad

      out._backward = _backward # saving the backward function
      return out


    # MULTIPLY
    def __mul__(self, other):
      other = other if isinstance(other, Value) else Value(other) # if other is primative integer convert it to value
      out = Value(self.data * other.data, (self, other), "*")

      def _backward():
        self.grad += other.data * out.grad # [local-grad * global-grad] chain-rule, local-grad = thing that is multiplying input it is constant derivative rule, f(x) =a*x f'(x)=a
        other.grad += self.data * out.grad

      out._backward = _backward
      return out

    # POWER
    def __pow__(self, other):
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = Value(self.data**other, (self,), f'**{other}')

      def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad # [local-grad * global-grad] = ax^(a-1) power rule * output expression gradient respect to loss

      out._backward = _backward
      return out

    # E^X
    def exp(self):
      x = self.data # input
      out = Value(math.exp(x), (self,), "exp")

      def _backward():
        self.grad += out.data * out.grad # [local-grad * global-grad] chain-rule, local-grad = e^x, global-grad = computation equation gradient e^x

      out._backward = _backward
      return out

    # TANH
    def tanh(self):    # computes tanH activation given self.data as input
      x = self.data
      t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
      out = Value(t, (self,), "tanh")

      def _backward():
        self.grad += (1-t**2) * out.grad # [local-grad * global-grad], local-grad is derivative of tanh-activation

      out._backward = _backward
      return out

    def __neg__(self): # -self, for subtraction
        return self * -1
    def __radd__(self, other): # other + self
        return self + other
    def __sub__(self, other): # self - other
        return self + (-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self, other): # other * self
        return self * other
    def __truediv__(self, other): # self / other
        return self * other**-1
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):
      topo = []
      visited = set()  # stors unqiue visited-nodes

      def build_topo(v):    # given root node-o
        if v not in visited:    # 3: if given-node ha snot been visited, mark is as visited
          visited.add(v)
          for child in v._prev:   # 1: iterate through all the children-ndoes of givne-node
            build_topo(child)     # 2: recursivly call function on all children-nodes
          topo.append(v)          # 3: after iterating chldren add given-root-node to topological-list
      build_topo(self)   # call function by first passing in o-root-node

      # go one variable at a time and apply the chain rule to get its gradient
      self.grad = 1
      for v in reversed(topo):
          v._backward()



class Neuron:

  def __init__(self, n_inputs): # Neuron(number of inputs to flowing into the node)
      # weights = tuple(), for every input-node create a value-weight-obj with ranomd initalized value, same for bias of this neuron
      self.w = list(Value(random.uniform(-1,1)) for _ in range(n_inputs))
      self.b = Value(random.uniform(-1, 1))
      self.n_inputs = n_inputs

  def __call__(self, x): # given input-x-arr computes w*x +b, this is called by doing n=Neuron(2) n([2.0, 3.0])
      # print(list(zip(self.w, x))) # zip of w/x pairs each elements in w with each input in x
      act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
      act = act.tanh()  # these vairables are Value-obj so call tanh() activation on weighted-sum
      return act 
      # weighted_sum = 0
      # for wi, xi in list(zip(self.w, x)):
      #     weighted_sum += (wi * xi)
      # weighted_sum += self.b
      # activation = weighted_sum.tanh()
      # return activation
  def parameters(self):
     return self.w + [self.b]  # concatanation of parameters
  def __repr__(self):
     return f"Node: inputs = {self.n_inputs} weights = [{self.get_weights_str()}]"
  def get_weights_str(self):
     output = ""
     for w in self.w:
        output += str(w.data)+", "
     return output


class Layer:
    # Layer(number of inputs, number of outputs or number of nodes in layer)
    def __init__(self, nin, nout):
        # for every node in current-layer create Neuron-obj for cur-layer passing in number of inputs for that neuron which in number of nodes in previous layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x): # given inputs to layer, 
        outs = [n(x) for n in self.neurons] # for each Neuron-obj in layer call forward-pass on each node adn store its output in list
        return outs   # return ouputs of each node in layer in a array
    
    def __repr__(self):
        string = "Layer:\t"
        for node in self.neurons:
           string += str(node)+"\n\t"
        return string
    def parameters(self):
       params = []  # collect all parameters of neurons in this layer in a single list-1D
       for neuron in self.neurons:
          params.extend(neuron.parameters())
       return params

class MLP:
    def __init__(self, nin, layer_sizes, learning_rate): # MLP(number of input-nodes into network, [array of sizes of all hidden and output layers])
      self.network_dimensions = [nin] + layer_sizes # combine them to create a list of all the sizes of layers
      # for every layer-size create a Layer passing in number of inputs to that layer as the cur-size and numbmer of nodes in that layer as the next-size
      self.layers = [Layer(self.network_dimensions[i], self.network_dimensions[i+1]) for i in range(len(layer_sizes))]
      self.learning_rate = learning_rate
      self.loss = 0

    def __call__(self, x): # for every layer in network compute the forward pass on each layer which reutrns list of outputs for cur-layer reuturn outputs of array of last layer
      for layer in self.layers:
        x = layer(x)
      return x
    def reset_grads(self):
       for p in self.parameters():
          p.grad = 0
  
    
    def forward_pass(self, x , show_preds=False):
       y_preds = []
       for example in x:
          y_preds.append(self.__call__(example))
       if show_preds: print("Y-Pred: "+str(y_preds))
       return y_preds
    
    def compute_loss(self, y, y_preds):
       loss_for_examples = []
       for label, pred in zip(y, y_preds):
          loss_for_examples.append((label - pred[0])**2)

       self.loss = sum(loss_for_examples)
       print("Loss: "+str(self.loss))

    def backward_pass(self):
      self.reset_grads()
      self.loss.backward()
      
    def update_parameters(self):
       for p in self.parameters():
          p.data += self.learning_rate * p.grad
    
    
    def parameters(self):
       params = []
       for layer in self.layers:
          params.extend(layer.parameters())
       return params
    
    def __repr__(self):
      string = ""
      for layer in self.layers:
         string += str(layer)+"\n"
      return string


def main():
    x = [                   # input-values each row is an example, each element in each row is input-value for that node in network
       [2.0, 3.0, -1.0],
       [3.0, -1.0, 0.5],
       [0.5, 1.0, 1.0],
       [1.0, 1.0, -1.0],
    ]
    y = [1.0, -1.0, -1.0, 1.0]  # labels or desired output values for each the 4 examples
    n = MLP(len(x[0]), [4, 4, 1], -0.05)

    y_preds = n.forward_pass(x)
    for i in range(20):
       y_preds = n.forward_pass(x)
       n.compute_loss(y, y_preds)
       n.backward_pass()
       n.update_parameters()
    print("Y-PRED (after gradient descent): "+str(y_preds))


    
    # SINGLE NEURON
    # x = [2.0, 3.0]     # creating 2 inputs into the neuron
    # n = Neuron(len(x)) # passing in number of inputs in Neuron-obj
    # print(n(x))   # calling that neuron computes its forward-pass and reuturns the output-Value-obj
    # print(n)

    # SINGLE LAYER
    # x = [2.0, 3.0]  # inputs into the layer or the output of nodes in previous layer
    # l = Layer(len(x), 3) # creating layer by passing number of nodes in previous layer and number of nodes in current layer
    # print(l(x))     # calling forward pass on that layer passing inputs or outputs of previous layer, which computes forward pass on each node in cur-layer and returns output of each node in cur-layer stores in list which are Value-obj
    # print(l)

    # NETWORK
    # x = [2.0, 3.0, -1.0]
    # n = MLP(len(x), [4, 4, 1]) # [3,4, 4, 1] is the architecture of the network
    # print("Outputs: "+str(n(x))) # passing in inputs into network to compute forward pass return list of outputs for output-layer which are Value-objs
    # print(n)

    # MANUAL OPTIMIZATION
    # x = [                   # input-values each row is an example, each element in each row is input-value for that node in network
    #    [2.0, 3.0, -1.0],
    #    [3.0, -1.0, 0.5],
    #    [0.5, 1.0, 1.0],
    #    [1.0, 1.0, -1.0],
    # ]
    # y = [1.0, -1.0, -1.0, 1.0]  # labels or desired output values for each the 4 examples
    # n = MLP(len(x[0]), [4, 4, 1], -0.05)
    # ypred = [n(example) for example in x]  # has 4 predictions bceause there 4 exampeles and 1 output-node, list of lists [[Value(data=0.6796846587854642, grad=0)], [Value(data=0.7896385140608211, grad=0)], [Value(data=0.44224397382376673, grad=0)], [Value(data=0.8391969511170917, grad=0)]]
    # print("Y: " + str(y))
    # print("Y-PRED: "+str(ypred))

    # print("---------------------------------------------------------------------------------------------------------")
    # loss = [(yout[0]-ygt)**2 for ygt, yout in zip(y, ypred)]  # for every label, prediction in pairing of y-dataset, predictions compute MSE loss
    # print("LOSS (each example):  "+str(loss))      # look at predictions and labels, more off the prediction higher the loss
    # loss = sum([(yout[0]-ygt)**2 for ygt, yout in zip(y, ypred)]) # sum MSE loss for each example
    # print("LOSS (sum): "+str(loss))
    # print("---------------------------------------------------------------------------------------------------------")

    # loss.backward()  # starting from root-node loss-Value-obj do a backward pass, when you print a
    # print("Parameters: "+str(len(n.parameters())))  # get all parametesr in network in list and see number of parameters
    # print("Random Neuron after backward has a gradient: "+str(n.layers[0].neurons[0].w[0]))
    
    # lr = -0.05
    # # number of iterations
    # for i in range(20):
    #   # compute forward
    #   ypred = [n(example) for example in x] 
    #   # compute loss, for pair every prediction and y-ground-truth, y-ouptut-label adn compute MSE for each example and sum
    #   loss = sum([(yout[0]-ygt)**2 for ygt, yout in zip(y, ypred)]) 
    #   # reset gradients
    #   for p in n.parameters():
    #      p.grad = 0
    #   # backprop acuumalte gradients through chain rule for this iteration
    #   loss.backward() 
    #   # iterate parmeters and update with leanring-rate and parameter gradient
    #   for p in n.parameters():
    #     p.data += lr * p.grad
    #   print("LOSS (after gradient descent)-"+str(i)+": " +str(loss))

    # print("Y: " + str(y))
    # print("Y-PRED (after gradient descent): "+str(ypred))
    # # if learning rate/iterations is too high it might over step and increase loss

    

main()

