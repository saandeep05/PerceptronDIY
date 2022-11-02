from DataPreprocessing import X_train, X_test, y_train, y_test
from Perceptron import Perceptron
from Metrics import Metrics

inp = X_train
inp['target'] = y_train

testing = X_test
testing['target'] = y_test

my_model = Perceptron(alpha=1e-4, epochs=10, theta=-0.05)
my_model.train(inp)
result, target = my_model.test(testing)

model_metrics = Metrics(target, result)
print("Efficiency: ", model_metrics.efficiency())
print("--------Confusion Matrix---------\n", model_metrics.confusion_matrix())
