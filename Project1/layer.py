from activation_functions import ActivationFunction
import numpy as np
from flags import VERBOSE


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
        self.bias_weights = bias_weights
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation = np.zeros(output_nodes)
        self.weighted_sums = np.zeros(output_nodes)

    def propagate(self, X: np.ndarray):
        self.weighted_sums = (self.weights @ X.T + self.bias_weights).T
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation

    def __repr__(self):
        return f"Layer({self.input_dim}, {self.output_dim}, activation={self.activation_function})"
