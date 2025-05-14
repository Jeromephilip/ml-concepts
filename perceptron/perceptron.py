    
import numpy as np
class Perceptron:
    def __init__(self, learning_rate=1.0, n_epochs=10):
        self.lr = learning_rate
        self.num_epochs = n_epochs
        self.w = None

    def train(self, X, y):
        X_pad = [x + [1.0] for x in X] # added padding to account for bias term (cleanup)
        X_pad = np.array(X_pad)
        y = np.array(y)
        '''

        Notes:
        n_samples runs over the entire list of spam messages
        n_features runs over the tokenized input of each message 
        
        '''

        n_samples, n_features = X_pad.shape

        self.w = np.zeros(n_features)

        for epoch in range(self.num_epochs):
            for i in range(n_samples):
                x_i = X_pad[i]
                y_i = y[i]
                score = np.dot(self.w, x_i)

                if y_i * score <= 0: # mistake has been found -> update
                    self.w += self.lr * y_i * x_i 

    def predict(self, x): # prediction for test set
        x = np.array(x + [1.0])
        score = np.dot(self.w, x)
        return 1 if score >= 0 else -1

    def predict_all(self, X):
        return [self.predict(x) for x in X]


