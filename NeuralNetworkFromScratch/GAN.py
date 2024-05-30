from NNFS import *


class GenerativeAdversarialNet:
    
    def __init__(self, X_train, Y_train, num_epochs, G_dims, D_dims, problem_type="regression"):
        self.X_train = X_train
        self.Y_train = Y_train # shape=(examples, output-nodes), each row is an examples, in each example there is a list where each element is an output node
        self.num_epochs = num_epochs
        self.G_dims = G_dims  # dimension of generative-model passed as array with num-input-nodes included [input-nodes, 1,2,3,4, output-nodes]
        self.D_dims = D_dims
        self.generator_model = None
        self.discriminator_model = None
        self.X_fake = []  # same shape as X-train and output of generator-model
        self.Y_fake = []  # np-arr of zeros with same shape as Y-train, 0 indicates fake
        self.Y_real = []  # np-arr of ones with same shape as Y-train, 1 indicates real
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

        self.Y_fake = np.zeros(X_train.shape)
        self.Y_real = np.ones(X_train.shape) # shape of [[e1], [e2], [0], [1]], binary classification label shape
    
    def generate_fake_data(self): # uses model-G to output fake-data
        fake_data = self.generator_model.predict(self.get_random_noise_vector())  # pass random noise-vector into generator-model to get inital generated fake-data
        # print(fake_data)
        # print(fake_data.shape) 
        self.X_fake = fake_data
        return fake_data
    
    def get_random_noise_vector(self):
        examples = X_train.shape[0] # print(X_train.shape) # (num-examples, num-input-nodes)
        input_nodes = X_train.shape[1]  # equal to model.input_size
        noise = np.random.randn(examples, input_nodes) # create random-noise vector with same shape as X-train, from normal distribution
        
        # print(noise)
        # print(noise.shape)
        return noise

    def train(self):
        # TBD: debug each training step at a time. Currently on discriminator model with real data cost converging to nan and constant values.
        # TBD: generate synthetic images
        num_iterations = 300
        for _ in range(self.num_epochs):
            self.generate_fake_data()
            print("\nTraining Discriminator [Real-Samples]...")
            # Step 1: train D on real data, input=real input, output=binary-classification-ones-labels
            self.discriminator_model.train(self.X_train, self.Y_real, epochs=num_iterations, learning_rate=0.001, batch_size=self.X_fake.shape[0], print_cost=True)
            # print("Training Discriminator [Fake-Samples]...")
            # # Step 2: train D on fake data, input=fake-data-generated-by=G, outupt=binary-classification-zero-labels
            # self.discriminator_model.train(self.X_fake, self.Y_fake, epochs=self.num_epochs, learning_rate=0.01, batch_size=self.X_fake.shape[0], print_cost=True)

            # print("\nTraining Generator...")
            # # Step 3: train G, input=random-noise, output=real-data-samples-so-it-can-replicate
            # self.generator_model.train(self.get_random_noise_vector(), self.Y_train, epochs=self.num_epochs, learning_rate=0.01, batch_size=self.X_fake.shape[0], print_cost=True)

        Y_pred = self.generator_model.predict(self.get_random_noise_vector())
        print(Y_pred[0])


if __name__ == "__main__":
    def generate_noisy_sine_data(num_samples):
        X = np.linspace(0, 2*np.pi, num_samples)
        Y = np.sin(X) + 0.1 * np.random.randn(num_samples)
        return X, Y
    num_samples = 500
    X_train, Y_train = generate_noisy_sine_data(num_samples)
    X_train = X_train.reshape(-1, 1)

    discriminator_dimensions = [1, 5,5, 1]
    generator_dimensions = [1, 5,5, 1] # 1 output nodes because sine-curve takes 1 x-value and outputs y-value
    num_epochs = 100
    gan = GenerativeAdversarialNet(X_train=X_train, Y_train=Y_train, num_epochs=1, G_dims=generator_dimensions, D_dims=discriminator_dimensions)
    
    # gan.discriminator_model.print_network_architecture()
    # gan.generator_model.print_network_architecture()
    
    gan.train()

    # print(Y_train.shape)