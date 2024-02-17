import numpy as np
class Optimizer_SGD:
    def __init__(self, features_size, gradFunction=None, learning_rate=0.00001):
        self.learning_rate = learning_rate
        self.theta = np.random.randn(features_size[0],features_size[1]) # was np.zeros((features_size, 1))
        self.gradFunction = gradFunction

    def update(self):
        g = self.gradFunction(self.theta)
        self.theta = self.theta - self.learning_rate * g

    def updateWith(self, x, c):
        g = self.gradFunction( x, self.theta, c)
        self.theta = self.theta - self.learning_rate * g
    def getTheta(self):
        return self.theta

