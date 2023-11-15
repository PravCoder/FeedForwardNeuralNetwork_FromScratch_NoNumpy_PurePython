import math
"""
Input Layer (0):  1 node
Hidden Layer (1): 2 nodes
Output Later (2): 1 node

Binary classification: even number is classified as zero, odd number is classified as one.
Dataset: each input scalar has 1 output label scalar.
Examples = 3

Z[layer][node][example]
A[layer][node][example]

W[layer][prev][cur]
Z = []
A = []
W = []
x = [0.1, 0.2, 0.3]
y = [1, 0, 1]
# w[la]
"""

# activations
def sigmoid(z):
    return 1/(math.exp(-z))
# input dataset
x1 = 0.1
x2 = 0.2
x3 = 0.3
# output dataset
y1 = 1
y2 = 0
y3 = 1

# initalize parameters
w111 = 0.018230189162016477
w112 = 0.018230189162016477
w211 = 0.018230189162016477 
w221 = 0.018230189162016477 
b11 = 0.01
b12 = 0.01
b21 = 0.02

# Layer 1
z111 = w111*x1 + b11
z112 = w111*x2 + b11
z113 = w111*x3 + b11

a111 = max(0, z111)
a112 = max(0, z112)
a113 = max(0, z113)

z121 = w112*x1 + b12
z122 = w112*x2 + b12
z123 = w112*x3 + b12

a121 = max(0, z121)
a122 = max(0, z122)
a123 = max(0, z123)

# Layer 2
z211 = (w211*a111 + w221*a121) + b21
z212 = (w211*a112 + w221*a122) + b21
z213 = (w211*a113 + w221+a123) + b21
a211 = 1 / (1+math.e**(-z211))
a212 = 1 / (1+math.e**(-z212))
a213 = 1 / (1+math.e**(-z213))

print(f'Layer: {1}')
print(f'Z111= {z111}')
print(f'Z112= {z112}')
print(f'Z113= {z113}\n')
print(f'A111= {a111}')
print(f'A112= {a112}')
print(f'A113= {a113}\n')
print(f'Z121= {z121}')
print(f'Z122= {z122}')
print(f'Z123= {z123}\n')
print(f'A121= {a121}')
print(f'A112= {a112}')
print(f'A123= {a123}\n')

print(f'Z211= {z211}')
print(f'Z212= {z212}')
print(f'Z213= {z213}\n')

print(f'A211= {a211}')
print(f'A212= {a212}')
print(f'A213= {a213}')
print("----------------------------")

# Loss:
y_hat1 = a211
y_hat2 = a212
y_hat3 = a213
loss = ((y1-y_hat1)**2 + (y2-y_hat2)**2 + (y3-y_hat3)**2) * 1/3
print(f'Loss = {loss}')
print("----------------------------")

# Backpropagation:

# example-1
a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
b = sigmoid(z211) * (1-sigmoid(z211))
c = a111
d = w211
e = x1
f = a121
g = w221
h = x1
def i1(z111):
    if z111 > 0:
        return 1
    else:
        return 0
def j1(z121):
    if z121 > 0:
        return 1
    else:
        return 0
dLdW211_1 = a*b*c
dLdW221_1 = a*b*f
dLdW111_1 = a*b*d*i1(z111)*e
dLdW112_1 = a*b*g*j1(z121)*h

# example-2
a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
b = sigmoid(z212) * (1-sigmoid(z212))
c = a111
d = w211
e = x2
f = a122
g = w221
h = x2
def i2(z112):
    if z112 > 0:
        return 1
    else:
        return 0
def j2(z122):
    if z122 > 0:
        return 1
    else:
        return 0
dLdW211_2 = a*b*c
dLdW221_2 = a*b*f
dLdW111_2 = a*b*d*i2(z112)*e
dLdW112_2 = a*b*g*j2(z122)*h

#example-3
a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
b = sigmoid(z213) * (1-sigmoid(z213))
c = a113
d = w211
e = x3
f = a123
g = w221
h = x3
def i3(z113):
    if z113 > 0:
        return 1
    else:
        return 0
def j3(z123):
    if z123 > 0:
        return 1
    else:
        return 0
dLdW211_3 = a*b*c
dLdW221_3 = a*b*f
dLdW111_3 = a*b*d*i3(z113)*e
dLdW112_3 = a*b*g*j3(z123)*h

print("Gradients:")
print("dLdW211_1: "+str(dLdW211_1))
print("dLdW221_1: "+str(dLdW221_1))
print("dLdW111_1: "+str(dLdW111_1)+"\n")

print("dLdW211_2: "+str(dLdW211_2))
print("dLdW221_2: "+str(dLdW221_2))
print("dLdW111_2: "+str(dLdW111_2)+"\n")

print("dLdW211_3: "+str(dLdW211_3))
print("dLdW221_3: "+str(dLdW221_3))
print("dLdW111_3: "+str(dLdW111_3)+"\n")


# update parameters:
