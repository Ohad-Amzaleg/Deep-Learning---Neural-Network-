import numpy as np
from Classes.Activation_Functions_Classes.activation_function import ActivationFunction


class ReluClass(ActivationFunction):
    def __init__(self):
        pass

    def compute(self, input):
        return np.maximum(0, input)

    def computeGradient(self, input):
        return np.where(input > 0, 1, 0)
