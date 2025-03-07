"""
Homework 1: Linear Models
"""

import warnings
import sys
import numpy as np
from scipy.special import softmax

def warning(message, category, filename, lineno, file=None, line=None):
    red = "\033[91m"
    reset = "\033[0m"
    formatted_message = f"{red}{category.__name__}: {message}{reset}"
    print(formatted_message, file=file if file is not None else sys.stderr)
warnings.showwarning = warning

class StandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def train_test_split(X, Y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_count = int(len(indices) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

class LinearClassifier:
    def __init__(self, learning_rate=0.01, n_epochs=1000, scaler='none', verbose=False):
        """
        Softmax classifier with gradient descent and negative log likelihood loss.
        """

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.scaler = scaler.lower()
        if self.scaler == 'standard':
            self.scaler_obj = StandardScaler()
        elif self.scaler == 'minmax':
            self.scaler_obj = MinMaxScaler()
        elif self.scaler == 'none':
            self.scaler_obj = None
        else:
            warnings.warn("Invalid scaler. Choose 'standard', 'minmax', or 'none'. Defaulting to 'none'.")
            self.scaler_obj = None
        self.weights = None  

    def _add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((X, bias), axis=1)

    def _one_hot(self, y, n_classes): # aka. kronecker delta :)
        one_hot = np.zeros((y.shape[0], n_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the classifier using gradient descent.
        
        Parameters:
          X: numpy array of shape (n_samples, n_features)
          y: numpy array of integer labels, shape (n_samples,)
          X_val, y_val: optional validation set to monitor loss.
        """

        if self.scaler_obj is not None:
            X = self.scaler_obj.fit_transform(X)
            if X_val is not None:
                X_val = self.scaler_obj.transform(X_val)
        
        X = self._add_bias(X)
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        
        y_one_hot = self._one_hot(y, n_classes)
        
        self.train_loss_history = []
        self.val_loss_history = []
        
        for epoch in range(self.n_epochs):
            scores = np.dot(X, self.weights)
            probabilities = softmax(scores, axis=1)
            loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-15), axis=1))
            self.train_loss_history.append(loss)
            grad = np.dot(X.T, (probabilities - y_one_hot)) / n_samples
            
            self.weights -= self.learning_rate * grad
            
            if X_val is not None and y_val is not None:
                X_val_bias = self._add_bias(X_val)
                y_val_one_hot = self._one_hot(y_val, n_classes)
                scores_val = np.dot(X_val_bias, self.weights)
                probabilities_val = softmax(scores_val, axis=1)
                val_loss = -np.mean(np.sum(y_val_one_hot * np.log(probabilities_val + 1e-15), axis=1))
                self.val_loss_history.append(val_loss)
            
            if self.verbose and epoch % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch}, training loss: {loss:.4f}, validation loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, training loss: {loss:.4f}")
    
    def predict_proba(self, X):
        if self.scaler_obj is not None:
            X = self.scaler_obj.transform(X)
        X = self._add_bias(X)
        scores = np.dot(X, self.weights)
        return softmax(scores, axis=1)
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)