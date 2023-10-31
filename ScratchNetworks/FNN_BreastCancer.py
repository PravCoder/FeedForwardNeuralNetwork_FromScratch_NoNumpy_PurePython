from FNN import *
from sklearn.model_selection import train_test_split
from sklearn import datasets


def main():
    layers_dims = [30, 2, 4, 3, 1] 
    train_x, train_y = process_data()
    nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, l2_regularization=False, binary_classification=True, multiclass_classification=False, optimizer="momentum", gradient_descent_variant="batch")
    nn.train()
    #nn.check_gradients()
    nn.evaluate_accuracy()
    iters = []
    for i in range(nn.num_iterations):
        if i%100 == 0 or i % 100 == 0 or i == nn.num_iterations - 1:
            iters.append(i)
    plt.plot(iters, nn.costs, label = "Cost", color="red")
    plt.xlim(0, nn.num_iterations) 
    plt.ylim(0.68, 0.70)
    plt.show()
def process_data():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    examples = len(X_train) # = len(Y_train), Y_train = [1,0,1,0,..,m]
    x_nodes = len(X_train[0])
    y_nodes = 1
    proccessed_X_train = []
    proccessed_Y_train = []
    for n in range(x_nodes):
        proccessed_X_train.append([])
        for e in range(examples):
            proccessed_X_train[n].append(X_train[e][n])
    for n in range(y_nodes):
        proccessed_Y_train.append([])
        for e in range(examples):
            proccessed_Y_train[n].append(Y_train[e])

    return proccessed_X_train, proccessed_Y_train

main()

# COST-STATUS: decreasing for all iterations
# GRADIEN-CHECKING: incorrect backpropagation. Difference: 1.0-0.5104000318407103
# FILE-PURPOSE: this is the saved version of FNN before the major backaporagation and gradient checking refactor
# NOTE: should be computing all gradients for each layer.


