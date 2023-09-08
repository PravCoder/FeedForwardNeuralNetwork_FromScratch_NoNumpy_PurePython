from FNN import FeedForwardNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt



def main():
    colors = ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey', "White", "Black"]
    # TRAIN NETWORK
    layers_dims = [3, 10, 11]
    train_x, train_y, total = proccess_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.075, 3000, multiclass_classification=True)
    nn.train()
    # SINGLE EXAMPLE PREDICTION
    predict_inputs = [[165], [140], [146]]        # R,G,B
    pred_i, predictions = nn.predict(predict_inputs, [])  # no ouptut labels provided for predicitng single value it is optional
    print("Inputs: [[R],[G],[B]]: " + str(predict_inputs))
    print("Prediction: " + str(colors[pred_i]))
    print("Pred-i: " + str(pred_i))
    print("Predications: " + str([predictions]))
    print("------------------")
    get_class_num_predictions(nn)



def get_class_num_predictions(nn):
    correct = {'Blue':0, 'Brown':0, 'Green':0, 'Pink':0, 'Yellow':0, 'Orange':0, 'Purple':0, 'Red':0, 'Grey':0, "White":0, "Black":0} # {color: num correct}
    total = {'Blue':0, 'Brown':0, 'Green':0, 'Pink':0, 'Yellow':0, 'Orange':0, 'Purple':0, 'Red':0, 'Grey':0, "White":0, "Black":0} # {color: num correct}

    file = open("datasets/colors.txt", "r")
    total_examples = 5052    # number of lines colors.txt
    colors = [[0 for _ in range(total_examples)] for _ in range(11)] # for each output-nodes intalize a list of 100 example to zero
    for m, line in enumerate(file):
        x = line.split(",")
        red = int(x[0])
        blue = int(x[1])
        green = int(x[2])
        c = x[3].strip()
        if c == "Blue":
            total["Blue"] += 1
            colors[0][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i == 0:
                correct["Blue"] += 1
        if c == "Brown":
            total["Brown"] += 1
            colors[1][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i == 1:
                correct["Brown"] += 1
        if c == "Green":
            total["Green"] += 1
            colors[2][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i == 2:
                correct["Green"] += 1
        if c == "Pink":
            total["Pink"] += 1
            colors[3][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i == 3:
                correct["Pink"] += 1
        if c == "Yellow":
            total["Yellow"] += 1
            colors[4][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  4:
                correct["Blue"] += 1
        if c == "Orange":
            total["Orange"] += 1
            colors[5][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  5:
                correct["Orange"] += 1
        if c == "Purple":
            total["Purple"] += 1
            colors[6][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  6:
                correct["Purple"] += 1
        if c == "Red":
            total["Red"] += 1
            colors[7][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  7:
                correct["Orange"] += 1
        if c == "Grey":
            total["Grey"] += 1
            colors[8][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  8:
                correct["Grey"] += 1
        if c == "White":
            total["White"] += 1
            colors[9][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  9:
                correct["White"] += 1
        if c == "Black":
            total["Black"] += 1
            colors[10][m] = 1
            pred_i, preds = nn.predict([[red], [blue], [green]])
            if pred_i ==  10:
                correct["Black"] += 1

                
    # Define the colors and their corresponding labels
    print("Total Examples: " + str(total))
    print("Correct Examples: " + str(correct))
    colors = ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey', "White", "Black"]
    color_labels = ['Blue', 'Brown', 'Green', 'Pink', 'Yellow', 'Orange', 'Purple', 'Red', 'Grey', "White", "Black"]

    # Simulate the number of correct predictions for each color (replace with your actual data)
    correct_predictions = list(correct.values())  # Example correct prediction counts

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(colors, correct_predictions, color=colors)

    # Add labels and title
    plt.xlabel('Colors')
    plt.ylabel('Number of Correct Predictions')
    plt.title('Total Examples: ' + str(total_examples))

    # Display the graph
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Ensure labels are not cut off
    plt.show()
    # BAD:
    # The graph does not provide information on the model's accuracy for each color class in relation to the total number of examples for that class. 
    # The scale of the y-axis is misleading even though the model got some red examples correct it looks as if it didn't get any correct. 

def proccess_data():
    total = {'Blue':0, 'Brown':0, 'Green':0, 'Pink':0, 'Yellow':0, 'Orange':0, 'Purple':0, 'Red':0, 'Grey':0, "White":0, "Black":0} # {color: num correct}
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
            total["Blue"] += 1
        if c == "Brown":
            colors[1][m] = 1
            total["Brown"] += 1
        if c == "Green":
            colors[2][m] = 1
            total["Green"] += 1
        if c == "Pink":
            colors[3][m] = 1
            total["Pink"] += 1
        if c == "Yellow":
            colors[4][m] = 1
            total["Yellow"] += 1
        if c == "Orange":
            colors[5][m] = 1
            total["Orange"] += 1
        if c == "Purple":
            colors[6][m] = 1
            total["Purple"] += 1
        if c == "Red":
            colors[7][m] = 1
            total["Red"] += 1
        if c == "Grey":
            colors[8][m] = 1
            total["Grey"] += 1
        if c == "White":
            colors[9][m] = 1
            total["White"] += 1
        if c == "Black":
            colors[10][m] = 1
            total["Black"] += 1
    
    inputs = [reds, blues, greens]
    return np.array(inputs), np.array(colors), total
main()
