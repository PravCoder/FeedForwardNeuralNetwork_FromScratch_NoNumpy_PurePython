from NNFS import *
import pandas as pd
from sklearn.datasets import make_classification


def main():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    train_x = np.array(X)
    train_y = np.array(y)
    print(train_y.shape)
    examples, num_input_nodes = train_x.shape

    model = NeuralNetwork()
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=5, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=1, activation=Sigmoid(), initializer=Initializers.glorot_uniform))
    
    model.setup(cost_func=Loss.BinaryCrossEntropy, input_size=num_input_nodes, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(train_x, train_y, epochs=900, learning_rate=0.01, batch_size=examples)

    Y_pred = model.predict(train_x)

    example_indx = 999
    print(f'Actual: {y[example_indx]}')
    print(f'Prediction: {Y_pred[example_indx]}')

main()