import numpy as np
import scipy.special

class NeuralNetwork():

    # To Do:
    #       - verify ReLU activation function
    #       - add functionality for more hidden layers

    # class constructor
    def __init__(self, inNodes, hiddenNodes, outNodes, learningRate, activation=None, weightType=None):
        
        # initialize node parameters
        self.inNodes = inNodes
        self.hiddenNodes = hiddenNodes
        self.outNodes = outNodes

        # initialize learning parameters
        self.alpha = learningRate
        self.activation = activation if activation else "Sigmoid"
        self.weightType = weightType if weightType else "NormalDist"

        # get initial weight matrices
        #       - "NormalDist" samples initial weights from normal distribution with 
        #       - stdev equal to 1 divided by the square root of incoming links to a node
        if self.weightType == "NormalDist":
            self.weightInHidden = np.random.normal(0.0, pow(self.inNodes, -0.5), (self.hiddenNodes, self.inNodes))
            self.weightHiddenOut = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outNodes, self.hiddenNodes))
        #       - "Random" generates random weight matrices with range (-0.5, 0.5)
        elif self.weightType == "Random":
            self.weightInHidden = np.random.rand(self.hiddenNodes, self.inNodes) - 0.5
            self.weightHiddenOut = np.random.rand(self.outNodes, self.hiddenNodes) - 0.5

        # set activation function
        if self.activation == "Sigmoid":
            self.activationFunction = lambda x: scipy.special.expit(x)
        elif self.activation == "ReLU":
            self.activationFunction = lambda x: max(0, x)

    # train neural network
    def train(self, input_list, label_list):

        # get input and label arrays
        inputs = np.array(input_list, ndim=2).T
        labels = np.array(label_list, ndim=2).T
        
        # calculate hidden layer input/output
        hidden_input = np.dot(self.weightInHidden, inputs)
        hidden_output = self.activationFunction(hidden_input)

        # calculate final layer input/output
        final_input = np.dot(self.weightHiddenOut, hidden_output)
        final_output = self.activationFunction(final_input)

        # get errors for backpropagation
        output_errors = labels - final_output
        hidden_errors = np.dot(self.weightHiddenOut.T, output_errors)

        # update hidden-output layer weights with gradient descent
        self.weightHiddenOut += self.alpha * np.dot((output_errors * final_output * (1.0 - final_output)), np.transpose(hidden_output))

        # update input-hidden layer weights with gradient descent
        self.weightInHidden += self.alpha * np.dot((hidden_errors * hidden_output * (1.0 - hidden_output)), np.transpose(inputs))

    # generate predictions for input array
    def predict(self, input_list):

        # get input array
        inputs = np.array(input_list, ndmin = 2).T
        
        # calculate hidden layer input/output
        hidden_inputs = np.dot(self.weightInHidden, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        # calculate final layer input/output
        final_inputs = np.dot(self.weightHiddenOut, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)
        
        return final_outputs


if __name__ == "__main__":

    test = NeuralNetwork(3, 3, 3, 0.3)
    print(test.predict([1.0, 0.5, -1.5]))