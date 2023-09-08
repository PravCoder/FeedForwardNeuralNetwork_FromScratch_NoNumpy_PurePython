from FNN import *
from sklearn.datasets import load_diabetes


def main():
    layers_dims = [10, 3, 3, 3, 1] 
    train_x, train_y = process_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, l2_regularization=False, binary_classification=False, multiclass_classification=False, regression=True, optimizer="gradient descent", gradient_descent_variant="batch")
    nn.train()
    #nn.evaluate_accuracy()
    iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, nn.num_iterations) # og on both is (0, 900)
    plt.ylim(0.68, 0.70)
    plt.show()

def process_data():
    data = load_diabetes()
    processed_x, processed_y = [], [[]]
    train_x = data.data
    train_y = data.target   # 1D-list that contains the singular output values for each example/patient
    
    examples = len(train_x)  # each row is a patient so each row is an example
    x_nodes = len(train_x[0]) # the elements of a row are the feature value for that example, length of a row is number of features
    y_nodes = 1

    for x in range(x_nodes):
        processed_x.append([])
        for e in range(examples):
            processed_x[x].append(train_x[e][x]/200)  # dividing by 100 to normaliztion to avoid gradient explosion to small or to large this is numerical instability
    for y in range(y_nodes):
        for e in range(examples):
            processed_y[0].append(train_y[e]/200)
    #print("X: " + str(len(processed_x[0]) == examples))
    #print("Y: " + str(len(processed_y[0]) == examples))
    return processed_x, processed_y

#process_data()
main()





"""

DIABETES DATASET INFORMATION:
In the imported dataset each row in data.data represents a different patient each column in that row
represents a feature.

TYPE: regression

INPUT FEATURES: total = 10
-Age
-Sex
-BMI
-Average Blood Pressure
-S1 (Total Serum Cholesterol)
-S2 (Low-Density Lipoproteins)
-S3 (High-Density Lipoproteins)
-S4 (Total Cholesterol / HDL Ratio)
-S5 (Log of Serum Triglycerides)
-S6 (Blood Sugar Level)

OUTPUT LABEL (Quantitative Measure of Disease Progression):
-Disease Progression: A quantitative measure of disease progression one year after baseline. This is the target variable that you aim to predict using the input features.

"""