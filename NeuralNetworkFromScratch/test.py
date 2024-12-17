import numpy as np
def ReLU_forward(Z):
        return np.maximum(0, Z)

def Sigmoid_forward(Z):
    Z = np.clip(Z, -500, 500)  
    return 1 / (1 + np.exp(-Z))

# Layer 1
print("LAYER 1")
X1 = np.array([[0, 0], 
               [0, 1], 
               [1, 0], 
               [1, 1]])

W1 = np.array([[-0.15711978,  1.91467883,  1.30879812],
               [-0.40953811, -1.25633144,  1.39848743]])

b1 = np.array([0.38129563, 0.31356892, 0.16569252])

Z1 = np.dot(X1, W1) + b1
print(f"{Z1=}, {Z1.shape}")
A1 = ReLU_forward(Z1)
print(f"{A1=}, {A1.shape}")

# Layer 2
print("LAYER 2")
W2 = np.array(
            [[ 1.30957982, -0.58575335,  0.27496437],
            [ 0.0615504 ,  0.38399249,  0.67944275],
            [ 1.27965209,  0.87114635, -0.02316647]])
b2 = np.array([0.53641851, 0.26092846, 0.36984623])
Z2 = np.dot(A1, W2) + b2
print(f"{Z2=}, {Z2.shape}")
A2 = ReLU_forward(Z2)
print(f"{A2=}, {A2.shape}")

# Layer 3
print("LAYER 3")
W3 = np.array(
            [[-0.69113884],
            [ 0.18692256],
            [-1.07356081]])
b3 = np.array([0.37719807])
Z3 = np.dot(A2, W3) + b3
print(f"{Z3=}, {Z3.shape}")
A3 = Sigmoid_forward(Z3)
print(f"{A3=}, {A3.shape}")