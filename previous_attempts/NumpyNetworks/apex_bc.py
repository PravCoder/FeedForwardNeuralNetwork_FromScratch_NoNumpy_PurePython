from apex import Net
from sklearn.model_selection import train_test_split
from sklearn import datasets


def process_data():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    examples = len(X_train) # = len(Y_train), Y_train = [1,0,1,0,..,m]
    x_nodes = len(X_train[0])
    y_nodes = 1
    proccessed_X_train = []
    proccessed_Y_train = []
    
    for e in range(examples):
        proccessed_X_train.append([])
        for x in range(x_nodes):
            proccessed_X_train[e].append(X_train[e][x])
    
    for e in range(examples):
        proccessed_Y_train.append([])
        for y in range(y_nodes):
            proccessed_Y_train[e].append(Y_train[e])

    # print(x_nodes)
    # print(len(proccessed_Y_train) == examples)
    return proccessed_X_train, proccessed_Y_train

if __name__ == "__main__":
    dimensions = [30, 10, 10, 5, 1]
    train_x, train_y = process_data()
    nn = Net(train_x, train_y, dimensions, 0.075, 50)
    nn.model()

