class Node:
    def __init__(self, id):
        self.id = id
        self.Z = []  # [m1, m2, m3]
        self.A = []  # [m1, m2, m3]
        self.weights_to = []  # [weights, connecting, to, node], weight in current layer of which node exists and its id

    def __repr__(self):
        return f'Node-{self.id}, weights-to: {self.weights_to}, z: {self.Z}, a: {self.A}\n'

    def __str__(self):
        return self.__repr__()
        

class Net:
    def __init__(self, x, y, layer_dims, num_epochs, learning_rate):
        self.x = x
        self.y = y
        self.m = len(self.x[0])
        self.layer_dims = layer_dims
        self.layers = []
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.create_network()

    def create_network(self):
        for i, layer_nodes in enumerate(self.layer_dims):
            self.layers.append([])  # add empty layer-list
            for n in range(layer_nodes):
                new_node = Node(n)
                if i > 0:
                    # iterate nodes in previous layer
                    for _ in range(len(self.layers[i-1])):
                        new_node.weights_to.append(0.018230189162016477)
                for m in range(self.m):
                    new_node.Z.append(0)
                    new_node.A.append(0)
                self.layers[i].append(new_node)

    def forward(self):
        for i in range(1, len(self.layers)):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            for n in cur_layer:
                for m in range(self.m):
                    # compute dot prod
                    pass
                    

    def __repr__(self):

        o = ""
        for i, layer in enumerate(self.layers):
            o += f'LAYER: {i}-----------------------\n'
            for node in layer:
                o += str(node)
        return o

x = [[1,2,3]]
y = [[0,1,0]]
nn = Net(x, y, [1,2,1], 1, 0.05)
print(nn)


