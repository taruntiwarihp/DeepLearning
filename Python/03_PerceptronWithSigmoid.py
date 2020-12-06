import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import operator
import json
np.random.seed(0)

class PerceptronWithSigmoid:
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def perceptron(self, x):
        return np.sum(self.w * x) + self.b
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))
    
    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x
    
    def grad_b(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)
    
    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False):
        # initialise the weights and bias
        self.w = np.random.randn(1, X.shape[1])
        self.b = 0
        if log or display_plot: 
            #accuracy = {}
            mse = {}
        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            dw, db = 0, 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.w -= learning_rate*dw
            self.b -= learning_rate*db
            
            if log or display_plot:
                Y_pred = self.predict(X)
                #Y_binarized = (Y >= SCALED_THRESHOLD).astype(np.int)
                #Y_pred_binarized = (Y_pred >= SCALED_THRESHOLD).astype(np.int)
                #accuracy[i] = accuracy_score(Y_binarized, Y_pred_binarized)
                mse[i] = mean_squared_error(Y, Y_pred)
        if log:
            #with open('perceptron_with_sigmoid_accuracy.json', 'w') as fp:
                #json.dump(accuracy, fp)
            with open('perceptron_with_sigmoid_mse.json', 'w') as fp:
                json.dump(mse, fp)
        if display_plot:
            #epochs_, accuracy_ = zip(*accuracy.items())
            #plt.plot(epochs_, accuracy_)
            #plt.xlabel("Epochs")
            #plt.ylabel("Train Accuracy")
            #plt.show()
            epochs_, mse_ = zip(*mse.items())
            plt.plot(epochs_, mse_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Error (MSE)")
            plt.show()
            
                    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.sigmoid(self.perceptron(x))
            Y.append(result)
        return np.array(Y)
