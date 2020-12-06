import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import operator
import json
np.random.seed(0)

class Perceptron:
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def perceptron(self, x):
        return np.sum(self.w * x) + self.b
    
    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False):
        # initialise the weights and bias
        self.w = np.random.randn(1, X.shape[1])
        self.b = 0
        if log or display_plot: 
            accuracy = {}
        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            for x, y in zip(X, Y):
                result = self.perceptron(x)
                if y == 1 and result < 0:
                    self.w += learning_rate*x
                    self.b += learning_rate
                elif y == 0 and result >= 0:
                    self.w -= learning_rate*x
                    self.b -= learning_rate
            if log or display_plot:
                Y_pred = self.predict(X)
                accuracy[i] = accuracy_score(Y, Y_pred)
        if log:
            with open('perceptron_accuracy.json', 'w') as fp:
                json.dump(accuracy, fp)
        if display_plot:
            epochs_, accuracy_ = zip(*accuracy.items())
            plt.plot(epochs_, accuracy_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Accuracy")
            plt.show()
                    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.perceptron(x)
            Y.append(int(result>=0))
        return np.array(Y)