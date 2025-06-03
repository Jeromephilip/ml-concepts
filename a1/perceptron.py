import numpy as np

class Perceptron:
    def __init__(self, learning_rate=1.0, n_epochs=10):
        self.lr = learning_rate
        self.num_epochs = n_epochs
        self.w = None
        self.b = 0
    
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0  # initial bias

        for epoch in range(self.num_epochs):
            print(f"=== Epoch {epoch + 1} ===")
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                score = np.dot(self.w, x_i) + self.b
                # print(f"Sample {i + 1}: x = {x_i.tolist()}, y = {y_i}, score = {score}")

                if y_i * score <= 0:
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i  # update bias
                    # print("→ Update made:")
                    # print(f"  New weights: {self.w}")
                    # print(f"  New bias: {self.b}")
                else:
                    pass
                    # print("→ No update needed")
            # print()


    def predict(self, x):
        x = np.array(x)
        score = np.dot(self.w, x) + self.b
        return 1 if score >= 0 else -1

    def predict_all(self, X):
        return [self.predict(x) for x in X]
