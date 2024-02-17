# Neural Network Project

Welcome to our Neural Network project! This project is implemented from scratch in Python, with a focus on understanding the fundamentals of neural networks. In this README, we'll provide an overview of the project structure, functionality, and how to use it.

## Project Structure

The project structure is organized as follows:

```
DeepLearning-NeuralNetwork/
│
├── Classes/
│   ├── ActivationFunctions/
│   │   ├── activation_functions.py
│   │   └── relu_class.py
│   │   └── sigmoid_class.py
│   │   └── tanh_class.py
│   │   └── softmax_class.py
│   ├── Layers/
│   │   ├── layer.py
│   │   ├── resNet_layer.py
│   |--Optimizer/
│   │   ├── optimizer_sgd_class.py
│   ├── Tests/
│   │   ├── test_gradient.py
│   │   └── test_jacobian.py


```

## Functionality

### neural_network.py

This file contains the main implementation of the neural network. It includes classes and functions for building, training, and using neural networks. Some key functionalities include:
- Definition of the neural network architecture
- Forward and backward propagation
- Loss computation
- Parameter updates using the SGD optimizer

### layer.py
    The layer.py file contains the implementation of the Layer class, which is the base class for all the layers in the neural network. It includes the forward and backward methods, which are used to compute the output of the layer and the gradients with respect to the input and parameters.

### activation_functions.py
    The activation_functions.py file contains the implementation of the ActivationFunction class, which is the base class for all the

### optimizer_sgd_class.py
    The optimizer_sgd_class.py file contains the implementation of the OptimizerSGD class, which is used to update the parameters of the neural network using the stochastic gradient descent (SGD) algorithm.

### resNet_layer.py
    The resNet_layer.py file contains the implementation of the ResNetLayer class, which is a specific type of layer used in the ResNet architecture. It includes the forward and backward methods, which are used to compute the output of the layer and the gradients with respect to the input and parameters.

### tests/

The `tests/` directory contains modules for testing the correctness of the implementation. Two main testing modules are included:
- `test_gradient.py`: This module performs gradient testing to verify that the gradients computed by the network are correct.
- `test_jacobian.py`: This module performs Jacobian testing to ensure that the Jacobian matrix computed by the network is accurate.

## How to Use

To use this neural network project, follow these steps:

1. Clone the repository to your local machine.
2. Import the `neural_network.py` file into your Python environment.
3. Create an instance of the `NeuralNetwork` class and define the architecture of the network.
4. Train the network using the `train` method, providing the training data and labels.
5. Use the trained network to make predictions on new data using the `predict` method.
6. Test the correctness of the implementation using the testing modules in the `tests/` directory.
7. Experiment with different network architectures, activation functions, and optimizers to see how they affect the performance of the network.
8. Enjoy using your own custom neural network implementation!