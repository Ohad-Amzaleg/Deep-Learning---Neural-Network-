import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function, learning_rate=0.015):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
        self.b = self.b / np.linalg.norm(self.b)
        self.learning_rate = learning_rate
        self.output = None
        self.activation_function = activation_function
        self.X = None

    def forward(self, X):
        self.X = X
        self.output = self.activation_function.compute(self.W @ X + self.b)

    def backward(self, grad=None):
        if grad is None:
            grad = np.eye(self.W.shape[1])
        self.W -= self.learning_rate * self.gradient_w(self.X)(grad)
        self.b -= self.learning_rate * self.gradient_b(self.X)(grad)
        return self.gradient_x(self.X)(grad)

    def get_weights(self):
        return self.W

    def get_output(self):
        return self.output

    def get_output_size(self):
        return self.W.shape[0]

    def get_input_size(self):
        return self.W.shape[1]

    def gradient_w(self, X):
        if (len(X.shape) == 1):
            X = X.reshape(-1, 1)
        return lambda V: (self.activation_function.computeGradient(self.W @ X + self.b) * V) @ X.T

    def gradient_x(self, X):
        if (len(X.shape) == 1):
            X = X.reshape(-1, 1)
        return lambda V: self.W.T @ (self.activation_function.computeGradient(self.W @ X + self.b) * V)

    def gradient_b(self, X):  # TODO: If you have bugs change asix=1 to axis=0 and the reshape
        if (len(X.shape) == 1):
            X = X.reshape(-1, 1)
        return lambda V: np.sum((self.activation_function.computeGradient(self.W @ X + self.b) * V), axis=1).reshape(-1,
                                                                                                                     1)
