# PerceptronDIY

## Perceptron Model
#### This project is an implementation of simple Perceptron model. A perceptron model can be created using the 'Perceptron' class with the defining parameters as per the user choice

### Train and Test
#### 'train' and 'test' methods takes in the input dataset which is expected to be in the following format:
##### First n columns are the features (X) and the last column is the target (y)

## Metrics
#### Metrics gives the measure of how good a perceptron model is. Creating an object of metrics class require parameters 'target' (desired output) and 'result' (derived output) so that the values can be compared and quality of the model is estimated.

### Efficiency
##### Returns the percentage of correct predictions of testing data

### Confusion matrix
##### Returns a matrix of (True, False), (Positives, Negatives) combination