from DataPreprocessing import X_train, X_test, y_train, y_test
from Perceptron import Perceptron
from Metrics import Metrics

inp = X_train
inp['target'] = y_train

testing = X_test
testing['target'] = y_test

# my_model = Perceptron(alpha=1e-4, epochs=10, theta=-0.05)
# my_model.train(inp)
# result, target = my_model.test(testing)

n = int(input('Number of combinations: '))
params = []
for i in range(n):
    print(f'Set {i+1}')
    a = float(input())
    ep = int(input())
    th = float(input())
    params.append([a, ep, th])

max_eff = 0
best_model = None

for i in range(n):
    an_model = Perceptron(alpha=params[i][0], epochs=params[i][1], theta=params[i][2])
    an_model.train(inp)
    res, tar = an_model.test(testing)
    an_metrics = Metrics(tar, res)
    curr_eff = an_metrics.efficiency()
    print(f'{i+1}: {curr_eff}')
    if float(curr_eff) > max_eff:
        max_eff = curr_eff
        best_model = an_model

print(max_eff)

# model_metrics = Metrics(target, result)
# print("Efficiency: ", model_metrics.efficiency())
# print("--------Confusion Matrix---------\n", model_metrics.confusion_matrix())
