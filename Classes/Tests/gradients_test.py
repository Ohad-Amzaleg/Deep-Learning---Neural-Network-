import unittest
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Classes.Activation_Functions_Classes.soft_max_class import SoftMaxLayer
from Classes.Layers.layer import Layer
from Classes.Activation_Functions_Classes.relu_class import ReluClass as Activation_ReLU
from Classes.Activation_Functions_Classes.tanh_class import TanhClass as Activation_Tanh
from Classes.Layers.resNet_layer import ResNetLayer as ResLayer

GMMDataset = loadmat('../../HW1_Data/GMMData.mat')
Xtrain = GMMDataset['Yt']
Xtest = GMMDataset['Yv']
Ctrain = GMMDataset['Ct'].T
Ctest = GMMDataset['Cv'].T


class GardientTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    # Initialize the layers and perform forward pass
    def setUp(self):
        self.layer1 = Layer(Xtrain.shape[0], 10, Activation_Tanh())
        self.layer2 = Layer(10, 15, Activation_Tanh())
        self.layer3 = Layer(15, 16, Activation_Tanh())
        self.output_layer = SoftMaxLayer(16, Ctrain.shape[1])
        self.epsilon = np.array([0.5 ** i for i in range(1, 8)])
        self.num_directions = 10
        self.u = np.random.randn(self.output_layer.get_output_size(), Xtrain.shape[1])

        self.layer1_resNet = ResLayer(Xtrain.shape[0], 10, Activation_Tanh())
        self.layer2_resNet = ResLayer(Xtrain.shape[0], 15, Activation_Tanh())
        self.output_layer_resNet = SoftMaxLayer(Xtrain.shape[0], Ctrain.shape[1])


    # Test the gradient of the hidden layer 3
    def test_gradient_f3(self):
        self.layer1.forward(Xtrain)
        self.layer2.forward(self.layer1.get_output())
        self.layer3.forward(self.layer2.get_output())
        self.output_layer.forward(self.layer3.get_output())

        grad_x = self.output_layer.compute_gradient_x(self.layer3.get_output(), Ctrain)  # l(x)
        grad = self.layer3.gradient_w(self.layer2.get_output())(grad_x)

        self.gradient_test(self.layer3, self.layer2.get_output(), grad,grad_x)

    # Test the gradient of the hidden layer 2
    def test_gradient_f2(self):
        self.layer1.forward(Xtrain)
        self.layer2.forward(self.layer1.get_output())
        self.layer3.forward(self.layer2.get_output())
        self.output_layer.forward(self.layer3.get_output())
        grad_x = self.output_layer.compute_gradient_x(self.layer3.get_output(), Ctrain)
        grad_x3 = self.layer3.gradient_x(self.layer2.get_output())(grad_x)
        grad = self.layer2.gradient_w(self.layer1.get_output())(grad_x3)

        self.gradient_test(self.layer2, self.layer1.get_output(), grad, grad_x3)


    def test_gradient_f1(self):
        self.layer1.forward(Xtrain)
        self.layer2.forward(self.layer1.get_output())
        self.layer3.forward(self.layer2.get_output())
        self.output_layer.forward(self.layer3.get_output())
        grad_x = self.output_layer.compute_gradient_x(self.layer3.get_output(), Ctrain)
        grad_x3 = self.layer3.gradient_x(self.layer2.get_output())(grad_x)
        grad_x2 = self.layer2.gradient_x(self.layer1.get_output())(grad_x3)
        grad = self.layer1.gradient_w(Xtrain)(grad_x2)

        self.gradient_test(self.layer1, Xtrain, grad,grad_x2)

    def test_gradient_f2_resNet(self):
        self.layer1_resNet.forward(Xtrain)
        self.layer2_resNet.forward(self.layer1_resNet.get_output())
        self.output_layer_resNet.forward(self.layer2_resNet.get_output())
        grad_x = self.output_layer_resNet.compute_gradient_x(self.layer2_resNet.get_output(), Ctrain)
        grad_w1 = self.layer2_resNet.gradient_w1(self.layer1_resNet.get_output())(grad_x)
        grad_w2 = self.layer2_resNet.gradient_w2(self.layer1_resNet.get_output())(grad_x)

        self.gradient_test_resNet(self.layer2_resNet, self.layer1_resNet.get_output(), grad_w1, grad_w2, grad_x)

    def test_gradient_f1_resNet(self):
        self.layer1_resNet.forward(Xtrain)
        self.layer2_resNet.forward(self.layer1_resNet.get_output())
        self.output_layer_resNet.forward(self.layer2_resNet.get_output())

        grad_x = self.output_layer_resNet.compute_gradient_x(self.layer2_resNet.get_output(), Ctrain)
        grad_x2 = self.layer2_resNet.gradient_x(self.layer1_resNet.get_output())(grad_x)
        grad_w1 = self.layer1_resNet.gradient_w1(Xtrain)(grad_x)
        grad_w2 = self.layer1_resNet.gradient_w2(Xtrain)(grad_x)

        self.gradient_test_resNet(self.layer1_resNet, Xtrain, grad_w1, grad_w2, grad_x2)

    def gradient_test(self, layer, X, grad, grad_x):
        num_of_var = 1
        zero_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]
        first_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]

        for _ in range(self.num_directions):
            # Generate random normalized direction
            d_vars = [np.random.randn(*var.shape) / np.linalg.norm(var) for var in [layer.W]]

            zero_order = [[] for _ in range(num_of_var)]
            first_order = [[] for _ in range(num_of_var)]

            f_x = layer.get_output()

            for eps in self.epsilon:
                f_x_w = np.tanh(np.dot(layer.W + eps * d_vars[0], X) + layer.b)
                f_x_vars = [f_x_w]

                # Calculate differences
                for i in range(num_of_var):
                    diff = abs(np.vdot(f_x_vars[i], grad_x) - np.vdot(f_x, grad_x))
                    zero_order[i].append(diff)

                    diff_jac = abs(
                        np.vdot(f_x_vars[i], grad_x) - np.vdot(f_x, grad_x) - np.vdot(grad, eps * d_vars[i]))
                    first_order[i].append(diff_jac)

            for i in range(num_of_var):
                zero_order_avg[i] += zero_order[i]
                first_order_avg[i] += first_order[i]

        for i in range(num_of_var):
            zero_order_avg[i] /= self.num_directions
            first_order_avg[i] /= self.num_directions

        dict = {0: 'w'}
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


    def gradient_test_resNet(self, layer, X, grad_w1, grad_w2, grad_x):
        num_of_var = 2
        zero_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]
        first_order_avg = [np.zeros_like(self.epsilon, dtype=float) for _ in range(num_of_var)]

        for _ in range(self.num_directions):
            # Generate random normalized direction
            d_vars = [np.random.randn(*var.shape) / np.linalg.norm(var) for var in
                      [layer.W1, layer.W2]]
            jac_var = [grad_w1, grad_w2]

            zero_order = [[] for _ in range(num_of_var)]
            first_order = [[] for _ in range(num_of_var)]

            f_x = layer.get_output()
            for eps in self.epsilon:
                f_x_w1 = np.dot(layer.W2, np.tanh(np.dot(layer.W1 + eps * d_vars[0], X) + layer.b1)) + X
                f_x_w2 = np.dot(layer.W2 + eps * d_vars[1], np.tanh(np.dot(layer.W1, X) + layer.b1))

                f_x_vars = [f_x_w1, f_x_w2]

                # Calculate differences
                for i in range(num_of_var):
                    diff = abs(np.vdot(f_x_vars[i], grad_x) - np.vdot(f_x, grad_x))
                    zero_order[i].append(diff)

                    diff_jac = abs(
                        np.vdot(f_x_vars[i], grad_x) - np.vdot(f_x, grad_x) - np.vdot(jac_var[i], eps * d_vars[i]))
                    first_order[i].append(diff_jac)

            for i in range(num_of_var):
                zero_order_avg[i] += zero_order[i]
                first_order_avg[i] += first_order[i]

        for i in range(num_of_var):
            zero_order_avg[i] /= self.num_directions
            first_order_avg[i] /= self.num_directions

        dict = {0: 'W1', 1: 'W2'}
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