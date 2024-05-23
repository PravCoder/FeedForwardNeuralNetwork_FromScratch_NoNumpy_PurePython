from NNFS import *


class GenerativeAdversarialNet:
    
    def __init__(self, num_epochs, G_dims, D_dims, problem_type="regression"):
        self.num_epochs = num_epochs
        self.G_dims = G_dims  # dimension of generative-model passed as array with num-input-nodes included [input-nodes, 1,2,3,4, output-nodes]
        self.D_dims = D_dims
        self.generator_model = None
        self.discriminator_model = None
        self.initialize_models()

    def initialize_models(self):
        self.generator_model = NeuralNetwork() # init empty nn-obj
        # iterate from 2nd element which is 1st layer to the output-layer
        for i in range(1, len(self.G_dims)): 
            # add layer-obj with number of nodes to generator-nn
            self.generator_model.add(Layer(num_nodes=self.G_dims[i], activation=ReLU(), initializer=Initializers.glorot_uniform))
        # specify model cost-function, number of input-nodes which is first elements in G-dims, and optimization-func
        self.generator_model.setup(cost_func=Loss.MSE, input_size=self.G_dims[0], optimizer=Optimizers.SGD(learning_rate=0.01))

        self.discriminator_model = NeuralNetwork()
        for i in range(1, len(self.D_dims)):
            self.discriminator_model.add(Layer(num_nodes=self.D_dims[i], activation=ReLU(), initializer=Initializers.glorot_uniform))
        # Loss function for model-D ia binary-cross-entropy because it predicts probality that given sample is real-data
        self.discriminator_model.setup(cost_func=Loss.BinaryCrossEntropy, input_size=self.D_dims[0], optimizer=Optimizers.SGD(learning_rate=0.01))


    def train(self):
        pass
        # train D

        # loss function for D/G
        
        # train G



if __name__ == "__main__":
    discriminator_dimensions = [1, 5,5, 1]
    generator_dimensions = [1, 5,5, 1]
    num_epochs = 100
    gan = GenerativeAdversarialNet(num_epochs=100, G_dims=generator_dimensions, D_dims=discriminator_dimensions)