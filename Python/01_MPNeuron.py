import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import operator
import json
np.random.seed(0)

class MPNeuron:
    
    def __init__(self):
        self.theta = None
        
    def mp_neuron(self, x):
        if sum(x) >= self.theta:
            return 1
        return 0
    
    def fit_brute_force(self, X, Y):
        accuracy = {}
        for theta in tqdm_notebook(range(0, X.shape[1]+1), total=X.shape[1]+1):
            self.theta = theta
            Y_pred = self.predict(X)
            accuracy[theta] = accuracy_score(Y, Y_pred)  
            
        sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
        best_theta, best_accuracy = sorted_accuracy[0]
        self.theta = best_theta
        
    def fit(self, X, Y, epochs=10, log=False, display_plot=False):
        self.theta = (X.shape[1]+1)//2
        if log or display_plot:
            accuracy = {}
        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            Y_pred = self.predict(X)
            tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
            if fp > fn and self.theta <= X.shape[1]:
                self.theta += 1
            elif fp < fn and self.theta >= 1:
                self.theta -= 1
            else:
                continue
                
            if log or display_plot:
                Y_pred = self.predict(X)
                accuracy[i] = accuracy_score(Y, Y_pred)
        if log:
            with open('mp_neuron_accuracy.json', 'w') as fp:
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
            result = self.mp_neuron(x)
            Y.append(result)
        return np.array(Y)
