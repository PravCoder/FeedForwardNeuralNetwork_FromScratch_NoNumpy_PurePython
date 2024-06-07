from NNFS import *
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import datasets

"""
Example: SYNTHETIC SKLEARN DATASET

def main():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    train_x = np.array(X)
    train_y = np.array(y)
    print(train_y.shape)
    examples, num_input_nodes = train_x.shape

    model = NeuralNetwork()
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=5, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=1, activation=Sigmoid(), initializer=Initializers.glorot_uniform))
    
    model.setup(cost_func=Loss.BinaryCrossEntropy, input_size=num_input_nodes, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(train_x, train_y, epochs=900, learning_rate=0.01, batch_size=examples)

    Y_pred = model.predict(train_x)

    example_indx = 999
    print(f'Actual: {y[example_indx]}')
    print(f'Prediction: {Y_pred[example_indx]}')

main()"""


def main():
    
    data = datasets.load_breast_cancer()
    num_features = len(data.feature_names)
    num_classes = len(data.target_names) # 2, but 1 output node for binary classification
    X = np.array(data.data)     # each row represents example, each element in row is a feature, lenght of row is 30. shape(examples, feautres)
    Y = np.array(data.target)   # 1D-list where each element is either 0/1 for each element. Shape
    num_examples = len(X)

    model = NeuralNetwork()
    model.add(Layer(num_nodes=30, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=1, activation=Sigmoid(), initializer=Initializers.glorot_uniform))
    
    model.setup(cost_func=Loss.BinaryCrossEntropy, input_size=num_features, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.print_network_architecture()
    model.train(X, Y, epochs=1000, learning_rate=0.75, batch_size=num_examples, print_cost=True)

    Y_pred = model.predict(X)  

    indx = 100
    print(f"Label {indx}th: {Y[indx]}")
    print(f"Predication {indx}th: {Y_pred[indx]}")

    indx = 101
    print(f"\nLabel {indx}th: {Y[indx]}")
    print(f"Predication {indx}th: {Y_pred[indx]}")

main()