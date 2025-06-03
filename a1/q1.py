import numpy as np
from perceptron import Perceptron

X = np.array([
    [-1,  1],
    [ 1, -1],
    [ 2,  2],
    [ 0,  3]
])

# Labels
y = np.array([-1, -1, 1, 1])

model = Perceptron(learning_rate=1.0, n_epochs=2)
model.train(X, y)

print(model.w)