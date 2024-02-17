import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    def __init__(self, C=None):
        self.X = None
        self.W = None
        self.C = C

    @abstractmethod
    def compute(self, input):
        pass

    @abstractmethod
    def computeGradient(self, input):
        pass
