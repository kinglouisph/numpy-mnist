import numpy as np
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data("mnist.npz")

_inputs = []
_testing = []
_outputs = []
_outputstesting = []

print("setting up")

oneOver256 = 1.0 / 256.0

#i am not a python programmer
for i in range(len(train_x)*1+0):
    a=[]
    for j in range(28):
        for k in range(28):
            a.append(train_x[i][j][k] * oneOver256)
    _inputs.append(a)
    b = [0,0,0,0,0,0,0,0,0,0]
    b[train_y[i]] = 1
    _outputs.append(b)
    if i % 5000 == 0:
        print(i)

inputs = np.array(_inputs)
outputs = np.array(_outputs)

print("done setting up")
print((inputs.shape, outputs.shape))




def activation(n):
    return (n > 0) * n

def activationd(n):
    return (n > 0) + 0

hiddenSize = 50

weights1 = np.random.rand(hiddenSize, 28*28)*2-1 #input to hidden
weights2 = np.random.rand(10, hiddenSize)*2-1 #hidden to output

alpha = 0.006

for iteration in range(200):
    error = 0
    for i in range(len(inputs)):
        layer1 = activation(np.matmul(weights1, inputs[i])) 
        layer2 = np.matmul(weights2, layer1)
        #print(layer1.shape, layer2.shape)
        
        layer2Deltas = outputs[i] - layer2
        layer1Deltas = np.multiply(np.matmul(weights2.T, layer2Deltas), activationd(layer1))
        
        error += np.sum(layer2Deltas ** 2)
        #for j in layer2Deltas:
        #    error += j ** 2


        #print((inputs[i].shape, layer1Deltas.shape, np.mat(inputs[i]).shape))
        weights1 += alpha * np.matmul(np.mat(inputs[i]).T, np.mat(layer1Deltas)).T
        weights2 += alpha * np.matmul(np.mat(layer2Deltas).T, np.mat(layer1))

        #print(layer1Deltas.shape, layer2Deltas.shape)

    if iteration % 10 == 0:
        print((iteration, error))


#i am not a python programmer

test1 = []


print("setting up testing db")
for i in range(len(test_x)):
    a=[]
    for j in range(28):
        for k in range(28):
            a.append(test_x[i][j][k] * oneOver256)
    test1.append(a)
    if i % 2000 == 0:
        print(i)

print("done setting up testing db")

correct = 0

tinputs = np.array(test1)

for i in range(len(tinputs)):
    layer1 = activation(np.matmul(weights1, tinputs[i])) 
    layer2 = np.matmul(weights2, layer1)
    
    a=0
    b=-99999999.0
    for j in range(10):
        if layer2[j] > b:
            b=layer2[j]
            a=j

    if a == test_y[i]:
        correct += 1
    
    if (i % 1000 == 0):
        print((i, correct / (i+1)))
print("testing data: " + str(correct) + "/"+str(len(tinputs))+", "+str(correct/len(tinputs)*100.0)+"%")


