import numpy as np

class Loss:
    def calculate(self, inputs, eoutput):
        losses = self.forward(inputs, eoutput)
        return np.mean(losses)

class Cat_cross_entropy(Loss):#categorical cross entropy

    def forward(self, inputs, eoutput):
        samples = len(inputs)
        clipped = np.clip(inputs, 1e-7, 1-1e-7)
        if len(eoutput.shape) == 1:#sparse encoding
            confidences = clipped[range(samples), eoutput]
        elif len(eoutput.shape) == 2:
            confidences = np.sum(clipped * eoutput, axis = 1)#one hot encoding
        return -np.log(confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:#conversion to 1 hot encoding
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs /samples#normalization







