from DataPreprocessing import X_train, X_test, y_train, y_test
from Perceptron import Perceptron
from Metrics import Metrics

inp = X_train
inp['target'] = y_train

testing = X_test
testing['target'] = y_test

# Number of combinations user wants to try
n = int(input('Number of combinations: '))
params = []

# input alpha, epochs, and activation function as combination
for i in range(n):
    print(f'Set {i+1}')
    a = float(input('Enter learning rate (alpha): '))
    ep = int(input('Enter number of epochs: '))
    act = input('Enter activation function: ')
    params.append([a, ep, act])

max_eff = 0
best_model = None
best_model_metrics = None

# create a perceptron model for each combination
# train and test it
# find the maximum efficiency and return the best model
for i in range(n):
    curr_model = Perceptron(alpha=params[i][0], epochs=params[i][1], activation=params[i][2])
    curr_model.train(inp)
    res, tar = curr_model.test(testing)
    curr_metrics = Metrics(tar, res)
    curr_eff = curr_metrics.efficiency()
    print(f'{i+1}: {curr_eff}')
    if float(curr_eff) > max_eff:
        max_eff = curr_eff
        best_model = curr_model
        best_model_metrics = curr_metrics

print(f'Maximum efficiency is: {max_eff} with alpha={best_model.alpha}, epochs={best_model.epochs}, activation={best_model.getActivationFunction()}')
print(f'___________________________________\n')
print(f'Confusion matrix of the best model')
print(f'___________________________________\n')
print(f'{best_model_metrics.confusion_matrix()}')
print(f'___________________________________\n')

# model_metrics = Metrics(target, result)
# print("Efficiency: ", model_metrics.efficiency())
# print("--------Confusion Matrix---------\n", model_metrics.confusion_matrix())
