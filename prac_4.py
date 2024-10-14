import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
deriv = lambda x: x * (1 - x)

# XOR dataset and random initialization
X, y = np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]])
np.random.seed(42)
w1, b1 = np.random.rand(2, 2), np.random.rand(1, 2)
w2, b2 = np.random.rand(2, 1), np.random.rand(1, 1)

# Training loop
for _ in range(10000):
    h = sigmoid(np.dot(X, w1) + b1)
    o = sigmoid(np.dot(h, w2) + b2)
    d2 = (y - o) * deriv(o)
    d1 = d2.dot(w2.T) * deriv(h)
    w2 += h.T.dot(d2) * 0.1; b2 += np.sum(d2, axis=0, keepdims=True) * 0.1
    w1 += X.T.dot(d1) * 0.1; b1 += np.sum(d1, axis=0, keepdims=True) * 0.1

# Output prediction
print(o)