from typing import List
import numpy as np
from layer import Layer
import os
import pickle
from activation_functions import Sigmoid, ReLU, Softmax, Tanh, Linear


class NeuralNetwork:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_path = os.path.join(os.getcwd(), config["dataset"])
        regularization = config["regularization"]
        self.alpha = config["regularization_weight"]
        self.lr = config["learning_rate"]
        self.loss_function = config["loss_function"]
        input_dimension = config["input_dimension"]
        layers = config["layers"]
        self.epochs = config["epochs"]

        if regularization == "lr1":
            self.regularization = lambda X: X / np.abs(X)
        elif regularization == "lr2":
            self.regularization = lambda X: X
        else:
            self.regularization = lambda _: 0

        self.layers: List[Layer] = []

        if self.loss_function == "MSE":
            self.loss = lambda y, y_pred: y - y_pred
        elif self.loss_function == "CEE":
            self.loss = lambda y, y_pred: y / y_pred

        for layer in layers:
            act_func_str = layer["activation_function"]
            self.output_activation = act_func_str
            if act_func_str == "Softmax":
                act_func = Softmax()
            elif act_func_str == "ReLU":
                act_func = ReLU()
            elif act_func_str == "Tanh":
                act_func = Tanh()
            elif act_func_str == "Linear":
                act_func = Linear()
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

    def load_data(self) -> None:
        """
        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.
        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(self.dataset_path, "rb") as file:
            data = pickle.load(file)
            self.x_train, self.y_train = (
                np.array(data["x_train"]),
                np.array(data["y_train"]),
            )
            self.x_test, self.y_test = (
                np.array(data["x_test"]),
                np.array(data["y_test"]),
            )
            if len(self.y_train.shape) < 2:
                self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))
                self.y_test = self.y_test.reshape((self.y_test.shape[0], 1))

    def propagate_forward(self, X: np.ndarray) -> np.ndarray:
        """Propagate an input forward to receive activation in NN.

        Args:
            X (np.ndarray): Input to propagate.

        Returns:
            np.ndarray: Output layer activation.
        """
        for layer in self.layers:
            X = layer.propagate(X)
        return X

    def propagate_backward(self, X: np.ndarray, y: np.ndarray):

        self.propagate_forward(X)  # Each layer memorizes by itself
        m = y.shape[0]
        deltas = [[] for _ in range(len(self.layers))]

        if self.output_activation != "Softmax":
            # Linearly independent output activation gives us a simplification of the Jacobian
            deltas[-1] = np.multiply(
                self.layers[-1].activation_function.df(self.layers[-1].activation),
                self.loss(y, self.layers[-1].activation),  # Derivative of MSE
            )
        else:  # Softmax jacobian is not fun :) Some tensor operations needed..
            deltas[-1] = np.einsum(
                "ijk,ik->ij",
                self.layers[-1].activation_function.df(self.layers[-1].activation),
                self.loss(y, self.layers[-1].activation),
            )

        # Softmax + Cross Entropy Loss coulda been simplified with this
        # deltas[-1] = y - self.layers[-1].activation

        for i in range(len(self.layers) - 2, -1, -1):
            # Simple jacobian since our hidden layer activation functions are linearly independent
            deltas[i] = np.multiply(
                self.layers[i].activation_function.df(self.layers[i].weighted_sums),
                (deltas[i + 1] @ self.layers[i + 1].weights),
            )

        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                prev_activation = X.T
            else:
                prev_activation = self.layers[i - 1].activation.T

            # Update weights:
            self.layers[i].weights += self.lr / m * (
                (deltas[i].T @ prev_activation.T)
            ) - self.alpha * self.regularization(self.layers[i].weights)
            self.layers[i].bias_weights += self.lr / m * (
                np.sum(deltas[i], 0).reshape((self.layers[i].output_dim, 1))
            ) - self.alpha * self.regularization(self.layers[i].bias_weights)

    def train(self):
        for i in range(self.epochs):
            print("Epoch: ", i)
            mini_batch = np.random.choice(self.x_train.shape[0], 200, replace=False)
            self.propagate_backward(self.x_train[mini_batch], self.y_train[mini_batch])

        pred = self.propagate_forward(self.x_test)

        n = len(self.y_test)
        correct = 0
        # pred = self.propagate_forward(self.x_test)[0]
        pred = self.propagate_forward(self.x_test)

        # pred = np.round(pred, 3)

        # print("Pred: ", pred)
        # print("test: ", self.y_test)

        if self.config["dataset"] != "data_breast_cancer.p":
            k = np.argmax(pred, axis=1)
            p = np.argmax(self.y_test, axis=1)
            print(k)
            print(p)
            print("Accuracy: ", np.sum(k == p) / k.shape[0])
        else:
            for i in range(n):
                # Predict by running forward pass through the neural network
                # Sanity check of the prediction
                assert 0 <= pred[i] <= 1, "The prediction needs to be in [0, 1] range."
                # Check if right class is predicted
                correct += self.y_test[i][0] == round(float(pred[i][0]))
            print("Accuracy: ", round(correct / n, 3))

    def __repr__(self):
        representation = "NeuralNetwork("
        for i, layer in enumerate(self.layers):
            representation += layer.__repr__()
            if i != len(self.layers) - 1:
                representation += ","
        representation += ")"
        return representation

