from NNFS import *
import pandas as pd
from sklearn.datasets import make_classification


def main():
    output_nodes = 3
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=output_nodes, n_clusters_per_class=1, random_state=42)
    y_one_hot_encoded = []
    for i, num in enumerate(y):
        if num == 0:
            temp = [0 for _ in range(output_nodes)]
            temp[0] = 1
            y_one_hot_encoded.append(temp)
        if num == 1:
            temp = [0 for _ in range(output_nodes)]
            temp[1] = 1
            y_one_hot_encoded.append(temp)
        if num == 2:
            temp = [0 for _ in range(output_nodes)]
            temp[2] = 1
            y_one_hot_encoded.append(temp)

    train_x = np.array(X)
    train_y = np.array(y_one_hot_encoded)
    
    examples, num_input_nodes = train_x.shape

    model = NeuralNetwork()
    model.add(Layer(num_nodes=30, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=5, activation=ReLU(), initializer=Initializers.normal))
    model.add(Layer(num_nodes=output_nodes, activation=Sigmoid(), initializer=Initializers.glorot_uniform))
    
    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=num_input_nodes, optimizer=Optimizers.SGD(learning_rate=0.01))
    # model.train(train_x, train_y, epochs=10000, learning_rate=0.01, batch_size=examples)

    # Y_pred = model.predict(train_x)

    # example_indx = 999
    # print(f'Actual Index: {y_one_hot_encoded[example_indx]}')
    # print(f'Prediction: {Y_pred[example_indx]}')

    print(train_x)
    print(train_x.shape)

main()

"""
X: This is a 2-dimensional array representing the features of the dataset. Each row corresponds to a single data sample, and each column represents a feature. In this specific case:
Rows: Each row in X represents a single data sample or observation.
Columns: Each column in X represents a feature of the dataset. There are a total of 20 features.
y: This is a 1-dimensional array representing the target labels or classes of the dataset. Each element in y corresponds to the class label of the corresponding row in X. In this specific case:
Elements: Each element in y represents the class label for the corresponding data sample in X.
"""
