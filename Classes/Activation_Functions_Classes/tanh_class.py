import numpy as np
from Classes.Activation_Functions_Classes.activation_function import ActivationFunction


class TanhClass(ActivationFunction):
    def __init__(self):
        super().__init__()

    def compute(self, input):
        """
        Computes the tanh activation for the given input and weights.

        :param X:Input data, shape (features, samples)
        :param W:Weights, shape (features, class)
        :return:Tanh activation output, shape (samples, class)

        """
        return np.tanh(input)


    def computeGradient(self, input):
        """
        Computes the gradient with respect to the weights.
        """
        return (1 - self.compute(input) ** 2)
