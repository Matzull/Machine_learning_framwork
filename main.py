import numpy as np
import Layers as ly
import Activation as Ac
import Loss as ls
import Optimizers as Op
import visualizer as Vs
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

####################################################
##Create dataset
X, y = spiral_data(samples = 50, classes = 3)
####################################################
#first layer creation (2 inputs/features, 3 neurons)
dense1 = ly.Layer_Dense(2, 64)
#ReLU activation function initialization
activation1 = Ac.ReLU()
#second layer creation
dense2 = ly.Layer_Dense(64, 3)
#CCE loss and Softmax initialized
loss_activation = Ac.Activation_Softmax_Loss_CatCrossEntropy()
#opitmization
optimizer = Op.Adam(learning_rate=0.05, decay=1e-5)
#data visualizer
visualizer = Vs.Visualizer('viz', subplots=2)
onlinedraw = True
#main loop
for epoch in range(100001):
    #first layer forward
    dense1.forward(X)
    #activation function for first layer
    activation1.forward(dense1.output)
    #second layer forward
    dense2.forward(activation1.output)
    #loss calculation
    loss = loss_activation.forward(dense2.output, y)
    #acuracy calculations
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')
        visualizer.updateData(loss, accuracy)
        if not epoch % visualizer.getInterval() and onlinedraw:
            #data visualization
            visualizer.draw()
    #backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    #parameter update
    optimizer.pre_update()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update()

#data visualization
visualizer.plot()

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')