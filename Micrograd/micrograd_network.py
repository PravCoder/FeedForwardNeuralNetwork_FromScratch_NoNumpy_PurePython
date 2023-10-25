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

      def _backward():
        self.grad += 1.0 * out.grad  # [local-grad * global-grad] chain-rule, global-grad = derivative of final output of expression respect to out
        other.grad += 1.0 * out.grad # output-grad is basically being copied into self-grad dand other-grad

      out._backward = _backward
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




class Neuron:

  def __init__(self, n_inputs): # Neuron(number of inputs to node, )
    # weights = tuple(), for every input-node create a value-weight-obj with ranomd initalized value, same for bias of this neuron
    self.w = (Value(random.uniform(-1,1)) for _ in range(n_inputs))
    self.b = Value(random.uniform(-1, 1))
    self.n_inputs = n_inputs

  def __call__(self, x): # given input-x-arr computes w*x +b, this is called by doing n=Neuron(2) n([2.0, 3.0])
    # print(list(zip(self.w, x))) # zip of w/x pairs each elements in w with each input in x
    act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
    act = act.tanh()
    return act 
    # weighted_sum = 0
    # for wi, xi in list(zip(self.w, x)):
    #     weighted_sum += (wi * xi)
    # weighted_sum += self.b
    # activation = weighted_sum.tanh()
    # return activation

    def __repr__(self):
        return f"inputs={self.inputs} weights={self.w}"

class Layer:
  # Layer(number of inputs, number of outputs or number of nodes in)
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x): # when layer object is called like Layer() givne inputs-x = []

    outs = [n(x) for n in self.neurons] # iterate nodes of layer and call forward pass on them
    return outs   # return ouputs of each node in layer in a array
  
  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
  def __init__(self, nin, layer_sizes): # (number of inputs int, array of sizes of each layer )
    self.network_dimensions = [nin] + layer_sizes # adds list of input integer to array of layer-sizes
    self.layers = [Layer(self.network_dimensions[i], self.network_dimensions[i+1]) for i in range(len(layer_sizes))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


def main():
    x = [2.0, 3.0]
    n = MLP(3, [4, 4, 1]) # 3-input-nodes, 2 hidden layers each with 4 nodes, 1-ouput-layer
    print(n)
    n(x)
    print("dimensions: "+str(n.network_dimensions))
    for i, layer in enumerate(n.layers):
        print("i: "+str(i)+", "+str(repr(layer)))
        for j, node in enumerate(layer.neurons):
            print("\tj: "+str(j)+" "+str(repr(node)))
main()

