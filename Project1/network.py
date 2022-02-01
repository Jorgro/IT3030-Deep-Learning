from typing import List
import numpy as np
from layer import Layer
import os
from activation_functions import Sigmoid, ReLU, Softmax


class NeuralNetwork:
    def __init__(
        self,
        regularization: str,
        regularization_weight: float,
        learning_rate: float,
        loss_function: str,
        input_dimension: int,
        output_activation: str,
        layers: List[dict],
    ):
        if regularization == "lr1":
            self.regularization = lambda X: X / np.abs(X)
        elif regularization == "lr2":
            self.regularization = lambda X: X
        else:
            self.regularization = lambda _: 0

        self.alpha = regularization_weight
        self.lr = learning_rate

        self.layers: List[Layer] = []

        for layer in layers:
            act_func_str = layer["activation_function"]
            if act_func_str == "Softmax":
                act_func = Softmax()
            elif act_func == "ReLU":
                act_func = ReLU()
            else:
                act_func = Sigmoid()

            weights = NeuralNetwork.get_range(layer["weight_range"], input_dimension, layer["size"])
            bias_weights = NeuralNetwork.get_range(layer["bias_range"], input_dimension, layer["size"])

            self.layers.append(Layer(input_dimension, layer["size"], act_func, weights, bias_weights))
            input_dimension = layer["size"]

        # self.input_layer: Layer = Layer(30, 25)
        # self.hidden_layers: List[Layer] = [Layer(25, 25)]
        # self.output_layer: Layer = Layer(25, 1)

    @staticmethod
    def get_range(method, in_dim, out_dim)->np.ndarray:
        if method == "glorot":
            mean = 0
            variance = 2.0 / (in_dim + out_dim)
            return np.normal(mean, variance, (out_dim, in_dim))
        elif isinstance(method, list):
            return np.random.uniform(method[0], method[1], (out_dim, in_dim))
        else:
            return np.random.uniform(-0.5, 0.5, (out_dim, in_dim))



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

        m = y.shape[0]
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
            - self.alpha * self.output_layer.weights / m
        )

        self.input_layer.weights += 0.01 * (
            deltas[0] @ X - self.alpha * self.input_layer.weights / m
        )

        layer = self.hidden_layers[0]
        layer.weights += 0.01 * (
            (deltas[1] @ self.input_layer.activation.T)
            - self.alpha * layer.weights / np.abs(layer.weights)
        )
        layer.bias_weights += 0.01 * (
            np.sum(deltas[1], 1).reshape((25, 1))
            - self.alpha * layer.bias_weights / np.abs(layer.bias_weights)
        )
        # print(np.sum(np.abs(layer.weights)))

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
