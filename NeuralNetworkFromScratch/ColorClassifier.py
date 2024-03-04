from NNFS import *




def main():

    x_train, y_train, examples, num_output_nodes, output_labels = process_data()

    model = NeuralNetwork()
    model.add(Layer(num_nodes=30, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=15, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=5, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=num_output_nodes, activation=Sigmoid(), initializer=Initializers.glorot_uniform))  # output-layer
 
    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=3, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(x_train, y_train, epochs=500, learning_rate=0.01, batch_size=examples)

    Y_pred = model.predict(x_train)

    print(f'Actual Color: {y_train[24]} {output_labels[list(y_train[24]).index(max(y_train[24]))]}')
    print(f'Predicted Color: {Y_pred[24]} {output_labels[list(Y_pred[24]).index(max(Y_pred[24]))]}')


def process_data():
    file = open("datasets/colors.txt", "r")
    features = []
    labels = []
    output_labels = []
    num_examples = 0

    for line in file:
        num_examples += 1
        line_elements = line.split(",")
        line_elements = [e.strip() for e in line_elements]
        if line_elements[3] not in output_labels:
            output_labels.append(line_elements[3])
        red = int(line_elements[0])
        green = int(line_elements[1])
        blue = int(line_elements[2])
        actual_color = line_elements[3]

        one_hot = [0,0,0,0,0,0,0,0,0,0,0]
        label_indx = output_labels.index(actual_color)
        features.append([red, blue, green])
        one_hot[label_indx] = 1
        labels.append(one_hot)

    print(f"Classes: {output_labels}")

    features_array = np.array(features)
    labels_array = np.array(labels)
    num_output_nodes = len(output_labels)

    return features_array, labels_array, num_examples, num_output_nodes, output_labels

if __name__ == "__main__":
    main()