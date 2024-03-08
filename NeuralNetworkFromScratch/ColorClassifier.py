from NNFS import *




def main():

    x_train, y_train, examples, num_output_nodes, output_labels = process_data()
    print(f'Examples: {examples}, Output Nodes: {num_output_nodes}, Output Labels: {output_labels}')

    model = NeuralNetwork()
    model.add(Layer(num_nodes=64, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=32, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=num_output_nodes, activation=Sigmoid(), initializer=Initializers.glorot_uniform))  # output-layer
 
    model.setup(cost_func=Loss.CategoricalCrossEntropy, input_size=3, optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(x_train, y_train, epochs=200, learning_rate=0.01, batch_size=examples)

    Y_pred = model.predict(x_train)

    example_indx = 345
    print(f'Actual Color: {y_train[example_indx]} {output_labels[list(y_train[example_indx]).index(max(y_train[example_indx]))]}')
    print(f'Predicted Color: {Y_pred[example_indx]} {output_labels[list(Y_pred[example_indx]).index(max(Y_pred[example_indx]))]}')


def process_data():
    file = open("datasets/colors.txt", "r")
    features = []
    labels = []
    output_labels = []
    num_examples = 0
    want_labels = ["Red", "Green", "Blue"]
    num_output_nodes = len(want_labels)

    for line in file:
        
        line_elements = line.split(",")
        line_elements = [e.strip() for e in line_elements]
        if line_elements[3] not in output_labels:
            output_labels.append(line_elements[3])
        red = int(line_elements[0])
        green = int(line_elements[1])
        blue = int(line_elements[2])
        actual_color = line_elements[3]

        if actual_color in want_labels:
            num_examples += 1
            one_hot = [0] * num_output_nodes
            label_indx = want_labels.index(actual_color)
            features.append([red, blue, green])
            one_hot[label_indx] = 1
            labels.append(one_hot)

    features_array = np.array(features)
    labels_array = np.array(labels)
    

    return features_array, labels_array, num_examples, num_output_nodes, want_labels

if __name__ == "__main__":
    main()