from scipy.io import loadmat
from Classes.neural_network import NeuralNetwork
from Classes.resNet_network import ResNetNetwork

GMMDataset = loadmat("HW1_Data/GMMData.mat")
Xtrain = GMMDataset["Yt"]
Xtest = GMMDataset["Yv"]
Ctrain = GMMDataset["Ct"].T
Ctest = GMMDataset["Cv"].T

# Neural Network Sample
# neural_network = NeuralNetwork(Xtrain.shape[0], Ctrain.shape[1])
# neural_network.train(100, Xtrain, Ctrain)

# ResNet Neural Network Sample
# resnet = ResNetNetwork(Xtrain.shape[0], Ctrain.shape[1])
# resnet.train(100, Xtrain, Ctrain)

