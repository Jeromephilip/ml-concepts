import numpy as np
from tensorflow.keras.datasets import mnist

class MulticlassPerceptron:
    def __init__(self, num_classes, input_dim, learning_rate=1.0, n_epochs=10):
        self.num_classes = num_classes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.W = np.zeros((num_classes, input_dim))

    def train(self, X, y):
        for epoch in range(self.n_epochs):
            print(f"=== Epoch {epoch + 1} ===")
            for i in range(X.shape[0]):
                x_i = X[i]
                y_i = y[i]

                scores = self.W @ x_i
                y_pred = np.argmax(scores)

                if y_pred != y_i:
                    self.W[y_i] += self.lr * x_i
                    self.W[y_pred] -= self.lr * x_i

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)
