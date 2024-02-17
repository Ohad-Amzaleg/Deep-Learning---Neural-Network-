import numpy as np
def leastSquaresLoss(A, y):
    return lambda theta: 0.5 * np.linalg.norm(A @ theta - y, 2) ** 2


def leastSquaresGrad(A, y):
    """
    :param A: Matrix of features
    :param y: Vector of labels
    :return: Gradient function
    """
    return lambda theta: A.T @ A @ theta - A.T @ y


def createMeanSquareData(noise=False):
    def func(x):
        return x ** 3 + 5 * x ** 2 + 1

    # Creating mean square error
    x_data = np.array([i for i in range(-20, 20)]) / 5
    noise_data = np.random.normal(0, 2, x_data.shape)
    y_data = (
        [func(x) for x in x_data] + noise_data
        if noise
        else [func(x) for x in x_data]
    )
    y_data = np.array(y_data).reshape(-1, 1)
    return x_data, y_data