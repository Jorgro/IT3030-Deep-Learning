from typing import List
import numpy as np
from layer import Layer
import os
import pickle
from activation_functions import Sigmoid, ReLU, Softmax


class NeuralNetwork:
    def __init__(self, config: dict):
        regularization = config["regularization"]
        self.alpha = config["regularization_weight"]
        self.lr = config["learning_rate"]
        loss_function = config["loss_function"]
        input_dimension = config["input_dimension"]
        output_activation = config["output_activation"]
        layers = config["layers"]

        if regularization == "lr1":
            self.regularization = lambda X: X / np.abs(X)
        elif regularization == "lr2":
            self.regularization = lambda X: X
        else:
            self.regularization = lambda _: 0

        self.layers: List[Layer] = []

        for layer in layers:
            act_func_str = layer["activation_function"]
            if act_func_str == "Softmax":
                act_func = Softmax()
            elif act_func_str == "ReLU":
                act_func = ReLU()
            else:
                act_func = Sigmoid()

            weights = NeuralNetwork.get_range(
                layer["weight_range"], input_dimension, layer["size"]
            )
            bias_weights = NeuralNetwork.get_range(
                layer["bias_range"], 1, layer["size"]
            )

            if "learning_rate" in layer.keys():
                lr = layer["learning_rate"]
            else:
                lr = self.lr

            self.layers.append(
                Layer(
                    input_dimension, layer["size"], act_func, weights, bias_weights, lr
                )
            )
            input_dimension = layer["size"]

    @staticmethod
    def get_range(method, in_dim, out_dim) -> np.ndarray:
        if method == "glorot":
            mean = 0
            variance = 2.0 / (in_dim + out_dim)
            return np.random.normal(mean, variance, (out_dim, in_dim))
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
        for layer in self.layers:
            X = layer.propagate(X)
        return X

    def propagate_backward(self, X: np.ndarray, y: np.ndarray):

        self.propagate_forward(X)  # Each layer memorizes by itself

        m = y.shape[0]
        deltas = [[] for _ in range(len(self.layers))]
        deltas[-1] = self.layers[-1].activation_function.df(
            self.layers[-1].weighted_sums
        ) * (y - self.layers[-1].activation)

        self.layers[-1].weights += self.lr * (
            deltas[-1] @ self.layers[-2].activation.T
            - self.alpha * self.regularization(self.layers[-1].weights)
        )

        for i in range(len(self.layers) - 2, -1, -1):
            deltas[i] = self.layers[i].activation_function.df(
                self.layers[i].weighted_sums
            ) * (self.layers[i + 1].weights.T @ deltas[i + 1])

            if i == 0:
                prev_activation = X
            else:
                prev_activation = self.layers[i - 1].activation.T

            self.layers[i].weights += self.lr * (
                (deltas[i] @ prev_activation)
                - self.alpha * self.regularization(self.layers[i].weights)
            )
            self.layers[i].bias_weights += self.lr * (
                np.sum(deltas[i], 1).reshape((self.layers[i].output_dim, 1))
                - self.alpha * self.regularization(self.layers[i].bias_weights)
            )

        ########
        """deltas[2] = self.output_layer.activation_function.df(
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
        ) """
        #######

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

    def __repr__(self):
        representation = "NeuralNetwork("
        for i, layer in enumerate(self.layers):
            representation += layer.__repr__()
            if i != len(self.layers) - 1:
                representation += ","
        representation += ")"
        return representation

