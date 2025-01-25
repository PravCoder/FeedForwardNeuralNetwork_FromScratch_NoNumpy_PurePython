from NNFS import *
from sklearn.datasets import load_digits

def one_hot_encode(Y, num_classes):
    encoded_Y = []
    all_labels = set(Y)
    template = [0 for _ in range(num_classes)]
    for label_indx in Y:
        encoded_example = template.copy()
        encoded_example[label_indx] = 1
        encoded_Y.append(encoded_example)
    
    return np.array(encoded_Y)

def main():
    data = load_digits()
    X = np.array(data.data) # each row is an exmaple-image, each element in row is a pixel-value, 64 cols in each row
    Y = np.array(data.target)  # 1D-list where each element in int between 0-9, denoting the class-label
    num_features = len(data.feature_names)
    num_classes = len(set(data.target))
    num_examples = len(X)  # number of rows in X
    all_labels = set(Y)
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}, {all_labels}")
    print(f"Number of examples: {num_examples}")
    Y_encoded = one_hot_encode(Y, num_classes)  # each row is label of example-image and is a encoded-on-hot vector where the correct class-indx is one.
    # print(Y_encoded)
    # print(Y_encoded.shape)

    model = NeuralNetwork()
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=5, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=3, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=num_classes, activation=Softmax(), initializer=Initializers.glorot_uniform))

    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=num_features, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(X, Y_encoded, epochs=500, learning_rate=0.01, batch_size=num_examples, print_cost=True)
    Y_pred = model.predict(X)  

    example_indx = 0
    max_indx = 0
    preds = [s for s in Y_pred[example_indx]]
    
    print(f"Label {example_indx}th example class index: {Y[example_indx]}")
    print(f"Predication {example_indx}th:  {preds.index(max(preds))}th class index. {Y_pred[example_indx]}")  # largest percentage in array is predicted-class

    

main()

"""
Dataset Info: 10 classes for each of the 10 digits. Each image is 8x8 pixels each image has 64 pixels. Each of these pixels is a feature, nunmber of nodes in input-layer is equal to the number of pixels in each digit-image-example
NUmber of examples ia 1797
"""