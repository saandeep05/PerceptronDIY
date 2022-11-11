import pandas as pd
import math


class Perceptron(object):
    def __init__(self, alpha=1e-5, epochs=10, theta=-0.05, activation='step', weights=[]):
        # member variables are initialized
        self.alpha = alpha
        self.epochs = epochs
        self.theta = theta
        self.__available_activations = ['step', 'relu', 'sigmoid']
        # Raise an error if activation function is not found
        if activation not in self.__available_activations:
            raise ValueError(f"No activation function named '{activation}'")
        self.__activation = activation
        self.weights = weights
        self.features = 0

    # Method to set activation function
    def setActivationFunction(self, activation):
        # Raise an error if activation function is not found
        if activation not in self.__available_activations:
            raise ValueError(f"No activation function named '{activation}'")
        self.__activation = activation

    # Method to get current activation function
    def getActivationFunction(self):
        return self.__activation

    # Method to get number of features
    def getFeatures(self):
        return self.features

    # Method to check available activation functions for this perceptron
    def getAvailableActivations(self):
        return self.__available_activations

    '''
    * Available activation functions are defined
    * Step, 
    * Rectified linear unit, 
    * Sigmoid
    '''
    def stepActivation(self, s):
        if s >= self.theta:
            return 1
        return 0

    def reluActivation(self, s):
        if s > 0:
            return 1
        return 0
    
    def sigmoidActivation(self, s):
        res = 1/(1+math.exp(-s))
        if res >= 0.5:
            return 1
        return 0

    # Method to select the necessary activation function
    def activationFunction(self, s):
        if self.__activation == 'step':
            return self.stepActivation(s)
        elif self.__activation == 'relu':
            return self.reluActivation(s)
        elif self.__activation == 'sigmoid':
            return self.sigmoidActivation(s)

    '''
    * Method to train the model using training data
    '''
    def train(self, inputs):
        # Every column except target is taken as a feature
        self.features = len(inputs.columns)-1

        # Set the initial weights
        if not self.weights:
            self.weights = [0 for i in range(self.features + 1)]  # +1 for bias

        # Perform iterations (epochs) over training data
        for i in range(self.epochs):
            for j in range(len(inputs)):
                # A row is one set of input
                # A row consists of features and target
                row = inputs.iloc[j]
                row = list(row)

                # Calculating sum = w1*x1 + w2*x2 + .... + b
                sum = 0
                for k in range(self.features):
                    sum += row[k] * self.weights[k]
                sum += self.weights[self.features]

                # derived output using activation function
                y = self.activationFunction(sum)

                # target (desired output) is acquired using indexing
                # Last column is target
                t = row[self.features]

                # error
                e = t - y

                # updating weights after each iteration of input
                for k in range(self.features):
                    self.weights[k] = self.weights[k] + self.alpha * row[k] * e
                self.weights[self.features] = self.weights[self.features] + self.alpha * e

    '''
    * Method to test the model using testing dataset
    '''
    def test(self, test_data):
        # result consists of derived values (predicted by model)
        result = []
        # target consists of desired values (correct values)
        target = []

        for i in range(len(test_data)):
            row = test_data.iloc[i]
            row = list(row)
            sum = 0
            for j in range(self.features):
                sum += row[j] * self.weights[j]
            sum += self.weights[self.features]

            y = self.activationFunction(sum)
            result.append(y)
            t = row[self.features]
            target.append(t)

        # at last return result and target as pd.Series dtype
        return pd.Series(result), pd.Series(target)
