from typing import List
import numpy as np
from activation_functions import Sigmoid, ActivationFunction


class Layer:
    def __init__(self, input_nodes, output_nodes):
        self.weights = np.ones((output_nodes, input_nodes))
        self.bias_weights = np.ones(output_nodes)
        self.activation_function: ActivationFunction = Sigmoid()
        self.activation = np.zeros(output_nodes)
        self.weighted_sums = np.zeros(output_nodes)

    def propagate(self, X: np.ndarray):
        self.weighted_sums = self.weights @ X.T + self.bias_weights
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation


class NeuralNetwork:
    def __init__(self):
        self.input_layer: Layer = Layer()
        self.hidden_layers: List[Layer] = []
        self.output_layer: Layer = Layer()

        self.weights = np.array([])

    def propagate_forward(self, X: np.ndarray) -> np.ndarray:
        """Propagate an input forward to receive activation in NN.

        Args:
            X (np.ndarray): Input to propagate.

        Returns:
            np.ndarray: Output layer activation.
        """

        x = self.input_layer.propagate(X)
        for layer in self.hidden_layers:
            x = layer.propagate(x)
        return self.output_layer.propagate(x)

    def propagate_backward(self, X: np.ndarray, y: np.ndarray):

        self.propagate_forward(X)  # Each layer memorizes by itself

        deltas = [[] for _ in range(self.L)]

        deltas[-1] = self.output_layer.activation_function.df(
            (self.output_layer.weighted_sums) @ (y - self.output_layer.activation)
        )

        for l in range(self.L - 2, 0, -1):
            layer = self.hidden_layers[l]
            deltas[l] = layer.activation_function.df(
                layer.weighted_sums @ (layer.weights * deltas[l])
            )

        deltas[0] = self.input_layer.activation_function.df(
            self.input_layer.weighted_sums @ (self.input_layer.weights * deltas[-1])
        )

        self.output_layer.weights += 0.01 * (self.output_layer.activation @ deltas[-1])
        self.input_layer.weights += 0.01 * (self.input_layer.activation @ deltas[0])

        for l in range(self.L - 2, 0, -1):
            layer = self.hidden_layers[l]
            layer.weights += 0.01 * (layer.activation @ deltas[l])

    def train(self):
        pass


if __name__ == "__main__":
    print("Running main application")
    l = Layer(3, 2)
    print(l.weights)
    print(l.propagate(np.array([[2, 3.5, 1], [3, 3.5, 7]])))

