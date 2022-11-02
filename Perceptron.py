import pandas as pd


class Perceptron(object):
    def __init__(self, alpha=1e-5, epochs=10, theta=-0.05, weights=[]):
        # self.features = features
        # self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.theta = theta
        self.weights = weights
        self.features = 0

    def setFeatures(self, features):
        self.features = features

    def getFeatures(self):
        return self.features

    def activationFunction(self, s):
        theta = -0.05
        if s >= theta:
            return 1
        return 0

    def train(self, inputs):
        self.features = len(inputs.columns)-1

        if not self.weights:
            self.weights = [0 for i in range(self.features + 1)]  # +1 for bias

        for i in range(self.epochs):
            for j in range(len(inputs)):
                # print(type(inputs))
                row = inputs.iloc[j]
                row = list(row)
                sum = 0
                for k in range(self.features):
                    sum += row[k] * self.weights[k]
                sum += self.weights[self.features]

                # derived output
                y = self.activationFunction(sum)
                # target (desired output) is acquired using indexing
                t = row[self.features]
                # error
                e = t - y

                # updating weights
                for k in range(self.features):
                    self.weights[k] = self.weights[k] + self.alpha * row[k] * e
                self.weights[self.features] = self.weights[self.features] + self.alpha * e

    def test(self, test_data):
        result = []
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
        return pd.Series(result), pd.Series(target)
