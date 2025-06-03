import numpy as np
from tensorflow.keras.datasets import mnist
from multiclassPerceptron import MulticlassPerceptron

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
x_test = x_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0

# Hyperparameters
num_classes = 10
input_dim = x_train.shape[1]
epochs = 10

# Train
model = MulticlassPerceptron(num_classes=num_classes, input_dim=input_dim, n_epochs=epochs)
model.train(x_train, y_train)

# Predict
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

# Evaluate
train_error = np.mean(train_preds != y_train)
test_error = np.mean(test_preds != y_test)

print(f"\nTrain error: {train_error:.4f}")
print(f"Test error: {test_error:.4f}")
