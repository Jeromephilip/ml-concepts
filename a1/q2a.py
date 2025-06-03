import numpy as np
from tensorflow.keras.datasets import mnist
from perceptron import Perceptron

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

num_classes = 10
input_dim = x_train.shape[1]
perceptrons = []

for label in range(num_classes):
    print(f"Training perceptron for class {label}")
    binary_labels = np.where(y_train == label, 1, -1)
    p = Perceptron(learning_rate=1, n_epochs=10)
    p.train(x_train, binary_labels)
    perceptrons.append(p)

def predict(X, perceptrons):
    X = np.array(X)
    all_scores = np.zeros((X.shape[0], len(perceptrons)))

    for i, p in enumerate(perceptrons):
        scores = X @ p.w + p.b
        all_scores[:, i] = scores
    
    return np.argmax(all_scores, axis = 1)

y_pred_train = predict(x_train, perceptrons)
y_pred_test = predict(x_test, perceptrons)

train_error = np.mean(y_pred_train != y_train)
test_error = np.mean(y_pred_test != y_test)

print(f"\nTrain error: {train_error:.4f}")
print(f"Test error: {test_error:.4f}")

