from activation_functions_1d import ActivationFunction
import numpy as np


class Layer:
    def __init__(
        self,
        input_nodes,
        output_nodes,
        activation_function: ActivationFunction,
        weights,
        bias_weights,
        learning_rate,
    ):
        self.input_dim = input_nodes
        self.output_dim = output_nodes
        self.weights = weights
        self.bias_weights = bias_weights.flatten()
        self.lr = learning_rate
        self.activation_function = activation_function

    def propagate(self, X: np.ndarray):
        self.weighted_sums = X @ self.weights + self.bias_weights
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation.copy()
