from FNN import *

def main():
    colors = ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey', "White", "Black"]
    # TRAIN NETWORK
    layers_dims = [3, 2, 11]
    train_x, train_y = proccess_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 50, l2_regularization=False, binary_classification=False, multiclass_classification=True, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch")
    nn.train()
    nn.evaluate_accuracy()
    # SINGLE EXAMPLE PREDICTION
    predict_inputs = [[255], [0], [0]]        # single example
    pred_i, predictions = nn.predict(predict_inputs, [], show_preds=True)  # no ouptut labels provided for predicitng single value it is optional
    print("Inputs-[[R],[G],[B]]: " + str(predict_inputs))
    print("Prediction: " + str(colors[pred_i]))
    # GRAPH COST BY ITERATIONS
    """iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, nn.num_iterations) # og on both is (0, 900)
    plt.ylim(0.40, 0.70)
    plt.show()"""
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
    return inputs, colors
main()

# OBSERVATIONS:
# - Accuracy decreased when iterations/archetecture were increased.
# -Math value domain error for increased examples and size of network
# DEBUG STRATIGIES:
# ()- Not enough examples for each class.
# ()- Decrease number of exmaples, and increase network size.


# NOTES:
# -Line 101 is where new data starts
# -Output Possible Labels: ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey'] each one is output node