from FNN import *



def generate_dataset(start, end):
    input_data = list(range(start, end + 1))
    output_labels = [0 if x % 2 == 0 else 1 for x in input_data]
    return input_data, output_labels

start_range = 1
end_range = 10

x_vals, y_vals = generate_dataset(start_range, end_range)
train_x, train_y = np.array([x_vals]), np.array([y_vals])

layers_dims = [1, 100, 1]
nn = FeedForwardNeuralNetwork(X=train_x,Y=train_y,dimensions=layers_dims, learning_rate=0.01, num_iterations=100, multiclass_classification=True, print_cost=False)
nn.train()

# print(train_x)
# print("\n"+str(train_y))

def test_prediction(preds):
    m = {0:"even", 1:"odd"}
    for pred in preds:
        pred_percent= nn.predict(np.array([[pred]]), None)[1][0] # (0, [0.51454900369935]) index of 1 is [0.5145] index of 0 is that value
        pred_binary = 1 if pred_percent > 0.5 else 0
        
        print(f'Prediction of {pred} is {m[pred_binary]}, {pred_percent}')
test_prediction([1,2,3,4,5,6,7,8,9,10,11,12,13])


# CLUES:
# - paramters not updating when printing, only some are updating rest are the same?