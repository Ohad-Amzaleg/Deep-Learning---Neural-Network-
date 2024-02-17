import numpy as np
from Classes.Activation_Functions_Classes.soft_max_class import SoftMaxLayer as SoftMaxLayer
from Classes.Activation_Functions_Classes.tanh_class import TanhClass as Activation_Tanh
from Classes.Layers.resNet_layer import ResNetLayer as ResLayer
class ResNetNetwork:
    def __init__(self, input_size, output_size):
        self.layer1 = ResLayer(input_size, 10, Activation_Tanh())
        self.layer2 = ResLayer(input_size, 15, Activation_Tanh())
        self.layer3 = ResLayer(input_size, 15, Activation_Tanh())
        self.layer4 = ResLayer(input_size, 10, Activation_Tanh())
        self.layer5 = ResLayer(input_size, 8, Activation_Tanh())
        self.output_layer = SoftMaxLayer(input_size,output_size)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]


    def feed_forward(self, input_data):
        output = input_data
        for layer in self.layers:
            layer.forward(output)
            output = layer.get_output()
        self.output_layer.forward(output)
        return output

    def back_propagation(self, input_data, target_data):
        self.output_layer.backward(input_data, target_data)
        grad = self.output_layer.compute_gradient_x(input_data, target_data)
        for layer in reversed(self.layers):
            new_grad = layer.backward(grad)
            grad = new_grad

    def train(self, epochs, input_data, target_data):
        loss = []
        for i in range(epochs):
            output=self.feed_forward(input_data)
            self.back_propagation(output, target_data)
            loss.append(self.output_layer.compute_loss(output, target_data))
            print(f"Epoch {i} Loss: {loss[i]}")


    def test(self, input_data, target_data):
        output = input_data
        for layer in self.layers:
            layer.forward(output)
            output = layer.get_output()
        self.output_layer.forward(output)
        print(
            "Test success percentage:",
            np.mean(np.argmax(self.output_layer.get_output(), axis=1) == np.argmax(target_data, axis=1))
            * 100,
            "%",
        )
