import numpy as np


class ResNetLayer:
    def __init__(self, input_size, middle_size, activation, learning_rate=0.01):
        self.W1 = np.random.randn(middle_size, input_size)
        self.W2 = np.random.randn(input_size, middle_size)
        self.learning_rate = learning_rate
        self.output = None
        self.activation = activation
        self.b1 = np.random.randn(middle_size, 1)
        self.X = None

    def forward(self, X):
        self.X = X
        self.output = X + self.W2 @ self.activation.compute(self.W1 @ X + self.b1)

    def get_weights(self):
        return self.W1, self.W2

    def get_output(self):
        return self.output

    def gradient_w1(self, X):
        return lambda v: (self.activation.computeGradient(self.W1 @ X + self.b1) * (self.W2.T @ v)) @ X.T

    def gradient_w2(self, X):
        return lambda v: v @ self.activation.computeGradient(self.W1 @ X + self.b1).T

    def gradient_x(self, X):
        def func(v):
            expression = self.W1.T @ (self.activation.computeGradient(self.W1 @ X + self.b1) * (self.W2.T @ v))
            I = np.eye(*expression.shape)
            return I + expression

        return func

    def gradient_b1(self, X):
        return lambda v: np.sum(self.activation.computeGradient(self.W1 @ X + self.b1) * (self.W2.T @ v),
                                axis=1).reshape(-1, 1)

    def backward(self, grad=None):
        if grad is None:
            grad = np.eye(self.W1.shape[0])
        self.W1 -= self.learning_rate * (self.gradient_w1(self.X)(grad))
        self.W2 -= self.learning_rate * (self.gradient_w2(self.X)(grad))
        self.b1 -= self.learning_rate * (self.gradient_b1(self.X)(grad))

        return self.gradient_x(self.X)(grad)

    def get_output_size(self):
        return self.W2.shape[0]
