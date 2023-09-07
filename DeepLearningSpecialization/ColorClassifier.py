from FNN import FeedForwardNeuralNetwork
import numpy as np


def main():
    colors = ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey', "White", "Black"]
    # TRAIN NETWORK
    layers_dims = [3, 10, 11]
    train_x, train_y = proccess_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.075, 3000, multiclass_classification=True)
    nn.train()
    # SINGLE EXAMPLE PREDICTION
    predict_inputs = [[38], [5], [64]]        # R,G,B
    pred_i, predictions = nn.predict(predict_inputs, [])  # no ouptut labels provided for predicitng single value it is optional
    print("Inputs-[[R],[G],[B]]: " + str(predict_inputs))
    print("Prediction: " + str(colors[pred_i]))
    print("Pred-i: " + str(pred_i))
    print("Predications: " + str([predictions]))

def proccess_data():
    file = open("datasets/colors.txt", "r")
    reds, blues, greens = [], [], []
    total_examples = 5052    # number of lines colors.txt
    colors = [[0 for _ in range(total_examples)] for _ in range(11)] # for each output-nodes intalize a list of 100 example to zero
    for m, line in enumerate(file):
        x = line.split(",")
        reds.append(int(x[0]))
        blues.append(int(x[1]))
        greens.append(int(x[2]))
        c = x[3].strip()
        if c == "Blue":
            colors[0][m] = 1
        if c == "Brown":
            colors[1][m] = 1
        if c == "Green":
            colors[2][m] = 1
        if c == "Pink":
            colors[3][m] = 1
        if c == "Yellow":
            colors[4][m] = 1
        if c == "Orange":
            colors[5][m] = 1
        if c == "Purple":
            colors[6][m] = 1
        if c == "Red":
            colors[7][m] = 1
        if c == "Grey":
            colors[8][m] = 1
        if c == "White":
            colors[9][m] = 1
        if c == "Black":
            colors[10][m] = 1
    inputs = [reds, blues, greens]
    return np.array(inputs), np.array(colors)
main()
