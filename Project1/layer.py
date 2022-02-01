from activation_functions import ActivationFunction
import numpy as np


class Layer:
    def __init__(
        self,
        input_nodes,
        output_nodes,
        activation_function: ActivationFunction,
        weights,
        bias_weights
    ):
        self.weights = weights
        self.bias_weights = bias_weights
        #self.weights = np.random.rand(output_nodes, input_nodes) - 0.5
        #self.bias_weights = np.ones((output_nodes, 1))
        self.activation_function = activation_function
        self.activation = np.zeros(output_nodes)
        self.weighted_sums = np.zeros(output_nodes)

    def propagate(self, X: np.ndarray):
        self.weighted_sums = self.weights @ X + self.bias_weights
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation
