import numpy as np


class SoftMaxLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, X):
        self.output = self.compute(X)

    def backward(self, X, C):
        self.W -= self.learning_rate * self.compute_gradient_w(X, C)
        return self.compute_gradient_x(X, C)

    def compute(self, X):
        """
         Computes the softmax activation for the given input and weights.

        :param X:Input data, shape (features, samples)
        :param W:Weights, shape (features, class)
        :return:Softmax activation output, shape (samples, class)

          """
        scores = X.T @ self.W
        exp_x = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def compute_loss(self, X, C):
        """
        Computes the cross-entropy loss for the softmax output.

        :param X: Input data, shape (features, samples).
        :param C: True class labels in one-hot encoded form, shape (samples,class).
        :return: float - Cross-entropy loss.

        """

        return np.sum(C * -np.log(self.compute(X))) / X.shape[1]

    def compute_gradient_w(self, X, C):
        """
          Computes the gradient of the cross-entropy loss with respect to the weights.

          :param X: Input data, shape (features, samples).
          :param C: True class labels in one-hot encoded form, shape (samples,class).
          :return: numpy.ndarray - Gradient of the loss with respect to the weights, shape (features, class).

          """
        # Compute softmax predictions
        P = self.compute(X)

        # Compute the gradient
        gradient = X @ (P - C)

        # Normalize by the number of samples
        gradient /= X.shape[1]

        return gradient

    def compute_gradient_x(self, X, C):
        """
            Computes the gradient of the cross-entropy loss with respect to the input data.
          :param X: Input data, shape (features, samples).
          :param C: True class labels in one-hot encoded form, shape (samples,class).
          :return: numpy.ndarray - Gradient of the loss with respect to the input data, shape (features, samples).
        """
        return (self.W @ (self.compute(X) - C).T) / X.shape[1]

    def get_output(self):
        return self.output

    def get_weights(self):
        return self.W

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_output_size(self):
        return self.output_size