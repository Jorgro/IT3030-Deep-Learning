from typing import List
import numpy as np
from activation_functions import Sigmoid, ActivationFunction
import os
import pickle


class Layer:
    def __init__(self, input_nodes, output_nodes):
        self.weights = np.random.rand(output_nodes, input_nodes) - 0.5
        self.bias_weights = np.ones((output_nodes, 1))
        self.activation_function: ActivationFunction = Sigmoid()
        self.activation = np.zeros(output_nodes)
        self.weighted_sums = np.zeros(output_nodes)

    def propagate(self, X: np.ndarray):
        self.weighted_sums = self.weights @ X + self.bias_weights
        self.activation = self.activation_function.f(self.weighted_sums)
        return self.activation


class NeuralNetwork:
    def __init__(self):
        self.input_layer: Layer = Layer(30, 25)
        self.hidden_layers: List[Layer] = [Layer(25, 25)]
        self.output_layer: Layer = Layer(25, 1)
        self.L = 1
        # self.weights = np.array([])

    def load_data(
        self, file_path: str = os.path.join(os.getcwd(), "data_breast_cancer.p")
    ) -> None:
        """
        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.
        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data["x_train"], data["y_train"]
            self.x_test, self.y_test = data["x_test"], data["y_test"]

    def propagate_forward(self, X: np.ndarray) -> np.ndarray:
        """Propagate an input forward to receive activation in NN.

        Args:
            X (np.ndarray): Input to propagate.

        Returns:
            np.ndarray: Output layer activation.
        """
        X = X.T
        x = self.input_layer.propagate(X)
        for layer in self.hidden_layers:
            x = layer.propagate(x)
        result = self.output_layer.propagate(x)
        # print("Res: ", result.shape)
        return result

    def propagate_backward(self, X: np.ndarray, y: np.ndarray):

        self.propagate_forward(X)  # Each layer memorizes by itself

        deltas = [[] for _ in range(3)]

        deltas[2] = self.output_layer.activation_function.df(
            self.output_layer.weighted_sums
        ) * (y - self.output_layer.activation)

        layer = self.hidden_layers[0]
        deltas[1] = layer.activation_function.df(layer.weighted_sums) * (
            self.output_layer.weights.T @ deltas[2]
        )

        deltas[0] = self.input_layer.activation_function.df(
            self.input_layer.weighted_sums
        ) * (self.hidden_layers[0].weights @ deltas[1])

        self.output_layer.weights += 0.01 * (
            deltas[2] @ self.hidden_layers[0].activation.T
        )

        self.input_layer.weights += 0.01 * (deltas[0] @ X)

        layer = self.hidden_layers[0]
        layer.weights += 0.01 * (deltas[1] @ self.input_layer.activation.T)

    def train(self):
        for i in range(500):
            print("Epoch: ", i)
            self.propagate_backward(self.x_train, self.y_train)

        pred = self.propagate_forward(self.x_test)

        n = len(self.y_test)
        correct = 0
        pred = self.propagate_forward(self.x_test)[0]
        pred = np.round(pred, 3)

        print("Pred: ", pred)
        print("test: ", self.y_test)
        for i in range(n):
            # Predict by running forward pass through the neural network
            # Sanity check of the prediction
            assert 0 <= pred[i] <= 1, "The prediction needs to be in [0, 1] range."
            # Check if right class is predicted
            correct += self.y_test[i] == round(float(pred[i]))
        print("Accuracy: ", round(correct / n, 3))


if __name__ == "__main__":
    print("Running main application")
    nn = NeuralNetwork()
    nn.load_data()
    nn.train()

