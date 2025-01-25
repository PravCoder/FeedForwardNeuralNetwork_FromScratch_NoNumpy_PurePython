from NNFS import *

class GANLoss:

    # Discriminator Loss Function
    class DLoss:

        @staticmethod
        def forward(AL, Y, D_out_real, D_out_fake):
            term1 = np.log(D_out_real)
            term2 = np.log(1 - D_out_fake)
            loss = -1/Y.shape[0] * np.sum(term1 + term2)
            return loss

        @staticmethod
        def backward(Y, D_out_real, D_out_fake, Y_fake, input_type="None"):
            m = Y.shape[0]
            dD_out_real = - (1 / D_out_real)
            dD_out_fake = 1 / (1 - D_out_fake)
            
            dA_prev_real = dD_out_real / m
            dA_prev_fake = dD_out_fake / m

            if input_type == "real":
                return dA_prev_real
            if input_type == "fake":
                return dA_prev_fake

            return None
    # Generator Loss Function
    class GLoss:

        @staticmethod
        def forward(AL, Y, D_out_fake):
            loss = -1/Y.shape[0] * np.sum(np.log(D_out_fake))
            return loss

        @staticmethod
        def backward(D_out_fake):
            m = D_out_fake.shape[0]
            dD_out_fake = -1 / D_out_fake

            dA_prev_fake = dD_out_fake / m

            return dA_prev_fake

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

        # GENERATOR
        self.generator_model = NeuralNetwork() # init empty nn-obj
        # iterate from 2nd element which is 1st layer to the output-layer-indx inclusive
        for i in range(1, len(self.G_dims)): 
            # add layer-obj with number of nodes to generator-nn
            self.generator_model.add(Layer(num_nodes=self.G_dims[i], activation=ReLU(), initializer=Initializers.glorot_uniform))
        self.generator_model.setup(cost_func=GANLoss.GLoss, input_size=self.G_dims[0], optimizer=Optimizers.SGD(learning_rate=0.01), is_gan_model="G")   # specify model cost-function, number of input-nodes which is first elements in G-dims, and optimization-func

        # DISCRIMINATOR
        self.discriminator_model = NeuralNetwork()
        # iterate from 2nd-element which is 2st layer skipping input-layer to layer before output-layer exclusive
        for i in range(1, len(self.D_dims)-1):
            self.discriminator_model.add(Layer(num_nodes=self.D_dims[i], activation=ReLU(), initializer=Initializers.glorot_uniform))
        self.discriminator_model.add(Layer(num_nodes=self.D_dims[len(self.D_dims)-1], activation=Sigmoid(), initializer=Initializers.glorot_uniform))

        self.discriminator_model.setup(cost_func=GANLoss.DLoss, input_size=self.D_dims[0], optimizer=Optimizers.SGD(learning_rate=0.01), is_gan_model="D")  # Loss function for model-D ia binary-cross-entropy because it predicts probality that given sample is real-data

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
        self.discriminator_model.print_network_architecture()
        self.generator_model.print_network_architecture()
        num_iterations = 1000
        for _ in range(self.num_epochs):
            self.generate_fake_data()
            D_out_real = self.discriminator_model.predict(X_train)
            D_out_fake = self.discriminator_model.predict(self.X_fake)
            # Step 1: train D on real data, input=real input, output=binary-classification-ones-labels
            print("\nTraining Discriminator [Real-Samples]...")
            self.discriminator_model.train(self.X_train, self.Y_real, epochs=num_iterations, learning_rate=0.001, batch_size=self.X_fake.shape[0], print_cost=True, D_out_real=D_out_real, D_out_fake=D_out_fake, Y_fake=self.Y_fake, input_type="real")

            # Step 2: train D on fake data, input=fake-data-generated-by=G, outupt=binary-classification-zero-labels
            print("\nTraining Discriminator [Fake-Samples]...")
            self.discriminator_model.train(self.X_fake, self.Y_fake, epochs=num_iterations, learning_rate=0.001, batch_size=self.X_fake.shape[0], print_cost=True, D_out_real=D_out_real, D_out_fake=D_out_fake, Y_fake=self.Y_fake, input_type="fake")

            # # Step 3: train G, input=random-noise, output=real-data-samples-so-it-can-replicate
            print("\nTraining Generator...")
            self.generator_model.train(self.get_random_noise_vector(), self.Y_train, epochs=num_iterations, learning_rate=0.01, batch_size=self.X_fake.shape[0], print_cost=True, D_out_real=D_out_real, D_out_fake=D_out_fake)

        # Y_pred = self.generator_model.predict(self.get_random_noise_vector())
        # print(Y_pred[0:5])
        # print(X_train[0:5])


if __name__ == "__main__":
    def generate_noisy_sine_data(num_samples):
        X = np.linspace(0, 2*np.pi, num_samples)
        Y = np.sin(X) + 0.1 * np.random.randn(num_samples)
        return X, Y
    num_samples = 400
    X_train, Y_train = generate_noisy_sine_data(num_samples)
    X_train = X_train.reshape(-1, 1)

    # Input: fake/real sine curve point. Output: prob of real/fake
    discriminator_dimensions = [2, 5,5, 1]
    # Input: random noise vector. Output: fake sine curve point
    generator_dimensions = [10, 10,10, 2]
    num_epochs = 1
    gan = GenerativeAdversarialNet(X_train=X_train, Y_train=Y_train, num_epochs=num_epochs, G_dims=generator_dimensions, D_dims=discriminator_dimensions)
    
    print(gan.Y_real)
    
    gan.train()

    Y_pred = gan.generator_model.predict(gan.get_random_noise_vector())
    print(Y_pred[0:10])


    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, Y_pred, label='Synthetic Data', color='red')
    plt.scatter(X_train, Y_train, label='Actual Data', color='blue')
    plt.title('GAN Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


    # NOTE:
    # combine the real and fake datapoints. The real features comes from real dataset and its labels are all 1, the fake features comes from what the generator outputs and fake labels are all 0. 
    # do generator.predict() to see fake samples.
    
