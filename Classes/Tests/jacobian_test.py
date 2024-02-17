import unittest
import numpy as np
from Classes.Layers.layer import Layer
from Classes.Activation_Functions_Classes.tanh_class import TanhClass as Activation_Tanh
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Classes.Layers.resNet_layer import ResNetLayer as ResLayer

GMMDataset = loadmat('../../HW1_Data/GMMData.mat')
Xtrain = GMMDataset['Yt']
Xtest = GMMDataset['Yv']
Ctrain = GMMDataset['Ct'].T
Ctest = GMMDataset['Cv'].T


class JacobinTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    # Initialize the layers and perform forward pass
    def setUp(self):
        self.layer1 = Layer(Xtrain.shape[0], 10, Activation_Tanh())
        self.resNet_layer = ResLayer(Xtrain.shape[0], 10, Activation_Tanh())

        # Perform forward pass
        self.layer1.forward(Xtrain)
        self.resNet_layer.forward(Xtrain)

        self.epsilon = np.array([0.5 ** i for i in range(1, 8)])
        self.num_directions = 10
        self.u = np.random.randn(self.layer1.get_output_size(), Xtrain.shape[1])
        self.u_resNet = np.random.randn(self.resNet_layer.get_output_size(), Xtrain.shape[1])

    def test_NN_layer(self):
        jac_w = self.layer1.gradient_w(self.layer1.X)
        jac_x = self.layer1.gradient_x(self.layer1.X)
        jac_b = self.layer1.gradient_b(self.layer1.X)

        self.JacobianTest(self.layer1, Xtrain, jac_w, jac_x, jac_b, )

    def test_resNet_layer(self):
        jac_w1 = self.resNet_layer.gradient_w1(self.resNet_layer.X)
        jac_w2 = self.resNet_layer.gradient_w2(self.resNet_layer.X)
        jac_x = self.resNet_layer.gradient_x(self.resNet_layer.X)
        jac_b = self.resNet_layer.gradient_b1(self.resNet_layer.X)

        self.JacobianResNetTest(self.resNet_layer, Xtrain, jac_w1, jac_w2, jac_x, jac_b)

    def JacobianTest(self, layer, X, jac_w, jac_x, jac_b):
        num_of_var = 3
        zero_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]
        first_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]

        for _ in range(self.num_directions):
            # Generate random normalized direction
            d_vars = [np.random.randn(*var.shape) / np.linalg.norm(var) for var in [layer.X, layer.W, layer.b]]
            jac_var = [jac_x, jac_w, jac_b]
            zero_order = [[] for _ in range(num_of_var)]
            first_order = [[] for _ in range(num_of_var)]

            f_x = layer.get_output()

            for eps in self.epsilon:
                f_x_d = np.tanh(np.dot(layer.W, X + eps * d_vars[0]) + layer.b)
                f_x_w = np.tanh(np.dot(layer.W + eps * d_vars[1], X) + layer.b)
                f_x_b = np.tanh(np.dot(layer.W, X) + layer.b + eps * d_vars[2])
                f_x_vars = [f_x_d, f_x_w, f_x_b]

                # Calculate differences
                for i in range(num_of_var):
                    diff = abs(np.vdot(f_x_vars[i], self.u) - np.vdot(f_x, self.u))
                    zero_order[i].append(diff)

                    diff_jac = abs(
                        np.vdot(f_x_vars[i], self.u) - np.vdot(f_x, self.u) - np.vdot(jac_var[i](self.u),
                                                                                      eps * d_vars[i]))
                    first_order[i].append(diff_jac)

            for i in range(num_of_var):
                zero_order_avg[i] += zero_order[i]
                first_order_avg[i] += first_order[i]

        for i in range(num_of_var):
            zero_order_avg[i] /= self.num_directions
            first_order_avg[i] /= self.num_directions

        dict = {0: 'X', 1: 'W', 2: 'B'}
        for i in range(num_of_var):
            plt.title("First order and zero order w.r.t " + dict[i])
            plt.plot(self.epsilon, zero_order_avg[i], label='Zero order approx')
            plt.plot(self.epsilon, first_order_avg[i], label='First order approx')
            plt.xlabel('Log Epsilon')
            plt.ylabel('Log Change')
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.show()

    def JacobianResNetTest(self, layer, X, jac_w1, jac_w2, jac_x, jac_b):
        num_of_var = 4
        zero_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]
        first_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]

        for _ in range(self.num_directions):
            # Generate random normalized direction
            d_vars = [np.random.randn(*var.shape) / np.linalg.norm(var) for var in
                      [layer.X, layer.W1, layer.W2, layer.b1]]
            jac_var = [jac_x, jac_w1, jac_w2, jac_b]

            zero_order = [[] for _ in range(num_of_var)]
            first_order = [[] for _ in range(num_of_var)]

            f_x = layer.get_output()
            for eps in self.epsilon:
                f_x_d = np.dot(layer.W2, np.tanh(np.dot(layer.W1, X + eps * d_vars[0]) + layer.b1)) + X + eps * d_vars[
                    0]
                f_x_w1 = np.dot(layer.W2, np.tanh(np.dot(layer.W1 + eps * d_vars[1], X) + layer.b1)) + X
                f_x_w2 = np.dot(layer.W2 + eps * d_vars[2], np.tanh(np.dot(layer.W1, X) + layer.b1))
                f_x_b = np.dot(layer.W2, np.tanh(np.dot(layer.W1, X) + layer.b1 + eps * d_vars[3])) + X
                f_x_vars = [f_x_d, f_x_w1, f_x_w2, f_x_b]

                # Calculate differences
                for i in range(num_of_var):
                    diff = abs(np.vdot(f_x_vars[i], self.u_resNet) - np.vdot(f_x, self.u_resNet))
                    zero_order[i].append(diff)

                    diff_jac = abs(
                        np.vdot(f_x_vars[i], self.u_resNet) - np.vdot(f_x, self.u_resNet) - np.vdot(
                            jac_var[i](self.u_resNet),
                            eps * d_vars[i]))
                    first_order[i].append(diff_jac)

            for i in range(num_of_var):
                zero_order_avg[i] += zero_order[i]
                first_order_avg[i] += first_order[i]

        for i in range(num_of_var):
            zero_order_avg[i] /= self.num_directions
            first_order_avg[i] /= self.num_directions

        dict = {0: 'X', 1: 'W1', 2: 'W2', 3: 'B1'}
        for i in range(num_of_var):
            plt.title("First order and zero order w.r.t " + dict[i])
            plt.plot(self.epsilon, zero_order_avg[i], label='Zero order approx')
            plt.plot(self.epsilon, first_order_avg[i], label='First order approx')
            plt.xlabel('Log Epsilon')
            plt.ylabel('Log Change')
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.show()
