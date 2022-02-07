from activation_functions import ActivationFunction
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
        no_bias,
    ):
        self.input_dim = input_nodes
        self.output_dim = output_nodes
        self.weights = weights
        self.bias_weights = bias_weights
        self.lr = learning_rate
        self.activation_function = activation_function
        self.activation = np.zeros(output_nodes)
        self.weighted_sums = np.zeros(output_nodes)
        self.no_bias = no_bias

    def propagate(self, X: np.ndarray):
        if self.no_bias:
            self.weighted_sums = (self.weights @ X.T).T
        else:
            self.weighted_sums = (self.weights @ X.T + self.bias_weights).T
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation

    def __repr__(self):
        return f"Layer({self.input_dim}, {self.output_dim}, activation={self.activation_function}, weights={np.average(np.abs(self.weights))}, bias={np.average(np.abs(self.bias_weights))}, no_bias={self.no_bias}, lr={self.lr})"
