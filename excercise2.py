import math
import matplotlib.pyplot as plt     # used to visualize the cost

learning_rate = 0.0075
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
z111 = 0
z112 = 0
z113 = 0
a111 = 0
a112 = 0
a113 = 0
z121 = 0
z122 = 0
z123 = 0
a121 = 0
a122 = 0
a123 = 0
# Layer 2
z211 = 0
z212 = 0
z213 = 0
a211 = 0
a212 = 0
a213 = 0

dLdW111_avr = 0
dLdW112_avr = 0
dLdW211_avr = 0
dLdW221_avr = 0

dLdW211_1 = 0
dLdW221_1 = 0
dLdW111_1 = 0
dLdW112_1 = 0

dLdW211_2 = 0
dLdW221_2 = 0
dLdW111_2 = 0
dLdW112_2 = 0

dLdW211_3 = 0
dLdW221_3 = 0
dLdW111_3 = 0
dLdW112_3 = 0

y_hat1 = 0
y_hat2 = 0
y_hat3 = 0

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

# TRAINING
def forward():
    global a213,a212,a211,a112,x1,z123,z113,x3,a123,a113,z213,x2,a122,y3,y2,y1,z122,z112,z212,z121,z111,z211,a121,a111,y_hat1, y_hat2, y_hat3, w111, w112, w211, w221, dLdW111_avr,dLdW112_avr,dLdW211_avr,dLdW221_avr,dLdW111_1,dLdW111_2,dLdW111_3,dLdW112_1,dLdW112_2,dLdW112_3,dLdW211_1,dLdW211_2,dLdW211_3,dLdW221_1,dLdW221_2,dLdW221_3
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
    return [a211, a212, a213]

def compute_loss(p1, p2, p3):
    global a213,a212,a211,a112,x1,z123,z113,x3,a123,a113,z213,x2,a122,y3,y2,y1,z122,z112,z212,z121,z111,z211,a121,a111,y_hat1, y_hat2, y_hat3, w111, w112, w211, w221, dLdW111_avr,dLdW112_avr,dLdW211_avr,dLdW221_avr,dLdW111_1,dLdW111_2,dLdW111_3,dLdW112_1,dLdW112_2,dLdW112_3,dLdW211_1,dLdW211_2,dLdW211_3,dLdW221_1,dLdW221_2,dLdW221_3
    y_hat1 = p1
    y_hat2 = p2
    y_hat3 = p3
    loss = ((y1-y_hat1)**2 + (y2-y_hat2)**2 + (y3-y_hat3)**2) * 1/3
    print("Loss: "+str(loss))
    return loss

def backward():
    global a213,a212,a211,a112,x1,z123,z113,x3,a123,a113,z213,x2,a122,y3,y2,y1,z122,z112,z212,z121,z111,z211,a121,a111,y_hat1, y_hat2, y_hat3, w111, w112, w211, w221, dLdW111_avr,dLdW112_avr,dLdW211_avr,dLdW221_avr,dLdW111_1,dLdW111_2,dLdW111_3,dLdW112_1,dLdW112_2,dLdW112_3,dLdW211_1,dLdW211_2,dLdW211_3,dLdW221_1,dLdW221_2,dLdW221_3
    # EXAMPLE-1
    a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
    b = sigmoid(z211) * (1-sigmoid(z211))
    c = a111
    d = w211
    e = x1
    f = a121
    g = w221
    h = x1
    dLdW211_1 = a*b*c
    dLdW221_1 = a*b*f
    dLdW111_1 = a*b*d*i1(z111)*e
    dLdW112_1 = a*b*g*j1(z121)*h
    # EXAMPLE-2
    a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
    b = sigmoid(z212) * (1-sigmoid(z212))
    c = a111
    d = w211
    e = x2
    f = a122
    g = w221
    h = x2
    dLdW211_2 = a*b*c
    dLdW221_2 = a*b*f
    dLdW111_2 = a*b*d*i2(z112)*e
    dLdW112_2 = a*b*g*j2(z122)*h
    # EXAMPLE-3
    a = ((y1-y_hat1) + (y2-y_hat2) + (y3-y_hat3)) * 1/3
    b = sigmoid(z213) * (1-sigmoid(z213))
    c = a113
    d = w211
    e = x3
    f = a123
    g = w221
    h = x3
    dLdW211_3 = a*b*c
    dLdW221_3 = a*b*f
    dLdW111_3 = a*b*d*i3(z113)*e
    dLdW112_3 = a*b*g*j3(z123)*h
    return [a213,a212,a211,a112,x1,z123,z113,x3,a123,a113,z213,x2,a122,y3,y2,y1,z122,z112,z212,z121,z111,z211,a121,a111,y_hat1, y_hat2, y_hat3, w111, w112, w211, w221, dLdW111_avr,dLdW112_avr,dLdW211_avr,dLdW221_avr,dLdW111_1,dLdW111_2,dLdW111_3,dLdW112_1,dLdW112_2,dLdW112_3,dLdW211_1,dLdW211_2,dLdW211_3,dLdW221_1,dLdW221_2,dLdW221_3]

def update():
    global a213,a212,a211,a112,x1,z123,z113,x3,a123,a113,z213,x2,a122,y3,y2,y1,z122,z112,z212,z121,z111,z211,a121,a111,y_hat1, y_hat2, y_hat3, w111, w112, w211, w221, dLdW111_avr,dLdW112_avr,dLdW211_avr,dLdW221_avr,dLdW111_1,dLdW111_2,dLdW111_3,dLdW112_1,dLdW112_2,dLdW112_3,dLdW211_1,dLdW211_2,dLdW211_3,dLdW221_1,dLdW221_2,dLdW221_3
    dLdW111_avr = (dLdW111_1 + dLdW111_2 + dLdW111_3) * 1/3     # average gradient caluclations over all examples
    dLdW112_avr = (dLdW112_1 + dLdW112_2 + dLdW112_3) * 1/3
    dLdW211_avr = (dLdW211_1 + dLdW211_2 + dLdW211_3) * 1/3
    dLdW221_avr = (dLdW221_1 + dLdW221_2 + dLdW221_3) * 1/3

    w111 = w111 - (learning_rate*dLdW111_avr)   # gradient descent
    w112 = w112 - (learning_rate*dLdW112_avr)
    w211 = w211 - (learning_rate*dLdW211_avr)
    w221 = w221 - (learning_rate*dLdW221_avr)


num_epochs = 100
costs = []
for _ in range(num_epochs):
    preds = forward()
    y_hat1, y_hat2, y_hat2 = preds[0], preds[1], preds[2]
    cur_loss = compute_loss(y_hat1, y_hat2, y_hat3)
    costs.append(cur_loss)
    backward()
    update()


iters = []
for i in range(num_epochs):
    iters.append(i)
plt.plot(iters, costs, label = "Cost", color="red")
plt.xlim(0, num_epochs+20)
plt.ylim(0.50, 0.51)
plt.show()

