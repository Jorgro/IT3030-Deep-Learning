from typing import List
import numpy as np
from layer import Layer
import os
import pickle
from activation_functions import Sigmoid, ReLU, Softmax, Tanh, Linear, LeakyReLU
import seaborn as sns
import matplotlib.pyplot as plt
from flags import VERBOSE


class NeuralNetwork:
    def __init__(self, config: dict):
        self.config = config
        self.mini_batch_size = config["batch_size"]
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
            self.loss_derivative = (
                lambda y, y_pred: y_pred - y
            )  # Mean squared derivative
            self.loss = lambda y, y_pred: 1 / y.shape[0] * np.sum((y - y_pred) ** 2)
        elif self.loss_function == "CEE":
            self.loss_derivative = lambda y, y_pred: np.where(
                y_pred != 0, -y / y_pred, 0
            )  # Cross entropy derivative
            self.loss = lambda y, y_pred: -1 / y.shape[0] * np.sum(y * np.log(y_pred))

        self.output_activation = config["type"]
        print("Output: ", self.output_activation)
        if self.output_activation == "Softmax":
            self.softmax = Softmax()

        for i, layer in enumerate(layers):
            act_func_str = layer["activation_function"]
            if act_func_str == "ReLU":
                act_func = ReLU()
            elif act_func_str == "LeakyReLU":
                act_func = LeakyReLU()
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

            # no_bias = len(layers) - 1 == i
            no_bias = False
            self.layers.append(
                Layer(
                    input_dimension,
                    layer["size"],
                    act_func,
                    weights,
                    bias_weights,
                    lr,
                    no_bias,
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
            self.x_val, self.y_val = (
                np.array(data["x_val"]),
                np.array(data["y_val"]),
            )
            self.x_test, self.y_test = (
                np.array(data["x_test"]),
                np.array(data["y_test"]),
            )
            if len(self.y_train.shape) < 2:
                self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))
                self.y_test = self.y_test.reshape((self.y_test.shape[0], 1))

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagate an input forward to receive activation in NN.

        Args:
            X (np.ndarray): Input to propagate.

        Returns:
            np.ndarray: Output layer activation.
        """
        for layer in self.layers:
            X = layer.propagate(X)
        if self.output_activation == "Softmax":
            return self.softmax.f(X)
        return X

    def backward_pass(self, X: np.ndarray, y: np.ndarray):
        if VERBOSE:
            print("INPUT: ", X)
        output = self.forward_pass(X)  # Each layer memorizes by itself
        if VERBOSE:
            print("OUTPUT: ", output)
            print("TARGET VALUES: ", y)
            print("ERROR: ", self.loss(y, output))
        m = y.shape[0]
        deltas = [[] for _ in range(len(self.layers))]

        # Calculate the first J_L_Z from network output
        if self.output_activation != "Softmax":
            # Linearly independent output activation gives a gradient matrix (examples x gradient vector)
            deltas[-1] = np.multiply(
                self.layers[-1].activation_function.df(self.layers[-1].weighted_sums),
                self.loss_derivative(y, self.layers[-1].activation),
            )
        else:  # Softmax jacobian is a tensor (when batching since gradient matrice x examples).
            deltas[-1] = np.einsum(
                "ijk,ik->ij",
                self.softmax.df(self.layers[-1].activation),
                self.loss_derivative(y, self.softmax.f(self.layers[-1].activation)),
            )  # J_L/Z = J_L/S * J_S_Z (roughly.. since tensors)

        # Softmax + Cross Entropy Loss coulda been simplified with this, but not done for being explicit.
        # deltas[-1] = y - self.softmax.f(self.layers[-1].activation)

        for i in range(len(self.layers) - 2, -1, -1):
            # Simple jacobian vectors since our hidden layer activation functions are linearly independent
            deltas[i] = np.multiply(
                self.layers[i].activation_function.df(self.layers[i].weighted_sums),
                (deltas[i + 1] @ self.layers[i + 1].weights),
            )
            # Calculate the J_L/Y = J_L/Z @ J_Z/Y, J_Z/Y = Diag_J_Z/Sum * W

        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                prev_activation = X.T
            else:
                prev_activation = self.layers[i - 1].activation.T

            # Update weights by calculating J_L_W/J_L_B (J_L_W = J_L_Z*J_Z_W_Hat)
            self.layers[i].weights -= self.layers[i].lr / m * (
                (deltas[i].T @ prev_activation.T)
            ) + self.alpha * self.regularization(self.layers[i].weights)
            self.layers[i].bias_weights -= self.layers[i].lr / m * (
                np.sum(deltas[i], 0).reshape((self.layers[i].output_dim, 1))
            ) + self.alpha * self.regularization(self.layers[i].bias_weights)

    def train(self):

        train_losses = []
        val_losses = []
        epochs = np.linspace(1, self.epochs, self.epochs)
        initial_lr = []
        for l in self.layers:
            initial_lr.append(l.lr)

        for i in range(self.epochs):
            print("Epoch: ", i + 1)
            mini_batch = np.random.choice(
                self.x_train.shape[0], self.mini_batch_size, replace=False
            )
            self.backward_pass(self.x_train[mini_batch], self.y_train[mini_batch])
            val_losses.append(self.loss(self.y_val, self.forward_pass(self.x_val)))
            train_losses.append(
                self.loss(self.y_train, self.forward_pass(self.x_train))
            )

            # Reduce learning over time
            m = 50
            for l, i_l in zip(self.layers, initial_lr):
                l.lr = i_l / (1 + i / m)
            #    print("LR: ", l.lr)

        pred = self.forward_pass(self.x_test)
        for l in self.layers:
            print("Layer activation: ", l.activation)
        if self.config["dataset"] != "data_breast_cancer.p":
            k = np.argmax(pred, axis=1)
            p = np.argmax(self.y_test, axis=1)
            print(k)
            print(p)
            print("Test accuracy: ", np.sum(k == p) / k.shape[0])
            print("Test cost: ", self.loss(self.y_test, pred))

        ax = sns.lineplot(x=epochs, y=train_losses)
        ax = sns.lineplot(x=epochs, y=val_losses, ax=ax)
        ax.set(xlabel="Epoch", ylabel="Error")
        ax.legend(
            [f"{self.loss_function}: Train loss", f"{self.loss_function}: Val loss"]
        )
        plt.show()

    def __repr__(self):
        representation = "NeuralNetwork("
        for i, layer in enumerate(self.layers):
            representation += layer.__repr__()
            if i != len(self.layers) - 1:
                representation += ","
        representation += ")"
        return representation

