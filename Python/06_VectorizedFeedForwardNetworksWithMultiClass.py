import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import operator
import json
np.random.seed(0)

class FF_MultiClass_InputWeightVectorised:

    def __init__(self, W1, W2):
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.B1 = np.zeros((1,2))
        self.B2 = np.zeros((1,4))

    def sigmoid(self, X):
        return 1.0/(1.0 + np.exp(-X))
    
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1,1)
    
    def forward_pass(self, X):
        self.A1 = np.matmul(X, self.W1) + self.B1 # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.sigmoid(self.A1) # (N, 2)
        self.A2 = np.matmul(self.H1, self.W2) + self.B2 # (N, 2) * (2, 4) -> (N, 4)
        self.H2 = self.softmax(self.A2) # (N, 4)
        return self.H2

    def grad_sigmoid(self, X):
        return X*(1-X)

    def grad(self, X, Y):
        self.forward_pass(X)
        m = X.shape[0]

        self.dA2 = self.H2 -Y #(N, 4) - (N, 4) -> (N, 4)

        self.dW2 = np.matmul(self.H1.T, self.dA2) # (2, N) * (N, 4) -> (2, 4)
        self.dB2 = np.sum(self.dA2, axis=0).reshape(1,-1) # (N, 4) -> (1, 4)
        self.dH1 = np.matmul(self.dA2, self.W2.T) # (N, 4) * (4, 2) -> (N, 2)
        self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)

        self.dW1 = np.matmul(X.T, self.dA1) # (2, N) * (N, 2) -> (2, 2) 
        self.dB1 = np.sum(self.dA1, axis=0).reshape(1, -1) # (N, 2) -> (1, 2)

    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):

        if display_loss:
            loss={}

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            self.grad(X, Y) # X -> (N, 2), Y -> (N, 4)

            m = X.shape[0]
            self.W2 -= learning_rate * (self.dW2/m)
            self.B2 -= learning_rate * (self.dB2/m)
            self.W1 -= learning_rate * (self.dW1/m) 
            self.B1 -= learning_rate * (self.dB1/m) 

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)

        if display_loss:
            plt.plot(np.fromiter(loss.values(), dtype = float))
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()