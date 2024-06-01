from NNFS import *
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import datasets

"""
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
    model.train(train_x, train_y, epochs=10000, learning_rate=0.01, batch_size=examples)

    Y_pred = model.predict(train_x)

    example_indx = 999
    print(f'Actual Index: {y_one_hot_encoded[example_indx]}')
    print(f'Prediction: {Y_pred[example_indx]}')

    # print(train_x)
    # print(train_x.shape)

main()"""

"""
X: This is a 2-dimensional array representing the features of the dataset. Each row corresponds to a single data sample, and each column represents a feature. In this specific case:
Rows: Each row in X represents a single data sample or observation.
Columns: Each column in X represents a feature of the dataset. There are a total of 20 features.
y: This is a 1-dimensional array representing the target labels or classes of the dataset. Each element in y corresponds to the class label of the corresponding row in X. In this specific case:
Elements: Each element in y represents the class label for the corresponding data sample in X.
"""

def one_hot_encode(Y):
    encoded_Y = []
    for label in Y:
        if label == 0:
            encoded_Y.append([1, 0, 0])
        if label == 1:
            encoded_Y.append([0, 1, 0])
        if label == 2:
            encoded_Y.append([0, 0, 1])

    return np.array(encoded_Y)

def main():

    data = datasets.load_iris()
    num_features = len(data.feature_names)
    num_classes = len(set(data.target))
    X = np.array(data.data) # each row is an example, each element in row is a input-node value
    Y = np.array(data.target) # 1D-list where each element is indx of class 0,1,2
    examples = X.shape[0]
    print(f"Number of Features: {num_features}, {data.feature_names}")
    print(f"Number of Classes: {num_classes}, {set(data.target)}")
    print(f"Examples: {examples}")
    # print(X)
    # print(X.shape)
    # print(Y)
    # print(Y.shape)
    Y_one_hot = one_hot_encode(Y) # each row is one-hot-encoded vector where true class is 1 and other classes are zero.
    # print(Y_one_hot)
    # print(Y_one_hot.shape)

    model = NeuralNetwork()
    
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=3, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=num_classes, activation=Softmax(), initializer=Initializers.glorot_uniform))

    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=num_features, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(X, Y_one_hot, epochs=1000, learning_rate=0.01, batch_size=examples, print_cost=True)

    Y_pred = model.predict(X)  
    

    example_indx = 0
    print(f"Label {example_indx}th example class index: {Y[example_indx]}")
    print(f"Predication {example_indx}th: {Y_pred[example_indx]}")  # largest percentage in array is predicted-class

    example_indx = 50
    print(f"\nLabel {example_indx}th example class index: {Y[example_indx]}")
    print(f"Predication {example_indx}th: {Y_pred[example_indx]}")

    example_indx = 100
    print(f"\nLabel {example_indx}th example class index: {Y[example_indx]}")
    print(f"Predication {example_indx}th: {Y_pred[example_indx]}")

    # Sometimes if learning rate is too high then numerical instability occurs, nan value occur in cost. And predition percentages will be 0,1 due to gradient clipping. 

main()


    