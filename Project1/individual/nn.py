from layer import Layer
import os
import numpy as np
from activation_functions_1d import (
    Softmax,
    ReLU,
    LeakyReLU,
    Linear,
    Tanh,
    Sigmoid,
    MSE,
    CEE,
)
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


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

        self.layers = []

        self.output_activation = config["type"]
        print("Output: ", self.output_activation)
        if self.output_activation == "Softmax":
            self.softmax = Softmax()

        if self.loss_function == "MSE":
            self.loss_function = MSE()
        else:
            print("ok")
            self.loss_function = CEE()

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

            self.layers.append(
                Layer(
                    input_dimension, layer["size"], act_func, weights, bias_weights, lr,
                )
            )
            input_dimension = layer["size"]

    @staticmethod
    def get_range(method, in_dim, out_dim, one_dim=False) -> np.ndarray:
        if method == "glorot":
            mean = 0
            variance = 2.0 / (in_dim + out_dim)
            v =  np.random.normal(mean, variance, (in_dim, out_dim))
        elif isinstance(method, list):
            v =  np.random.uniform(method[0], method[1], (in_dim, out_dim))
        else:
            v =  np.random.uniform(-0.5, 0.5, (in_dim, out_dim))
        if one_dim:
            return v.flatten()
        return v

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
            # self.x_val, self.y_val = (
            #     np.array(data["x_val"]),
            #     np.array(data["y_val"]),
            # )
            self.x_test, self.y_test = (
                np.array(data["x_test"]),
                np.array(data["y_test"]),
            )
            if len(self.y_train.shape) < 2:
                self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))
                self.y_test = self.y_test.reshape((self.y_test.shape[0], 1))

    def forward_pass(self, x):
        for l in self.layers:
            x = l.propagate(x)

        if self.output_activation == "Softmax":
            return self.softmax.f(x)
        return x

    def backward_pass(self, x, y):
        y_p = self.forward_pass(x)
        #print("Y: ", y)
        #print("Y_p: ", y_p)

        J_L_Z = self.loss_function.df(y, y_p)

        # Propagate through the weird softmax "layer"
        if self.output_activation == "Softmax":
            J_S_Z = self.softmax.df(y_p)
            J_L_Z = J_L_Z @ J_S_Z
        #print(J_L_Z)

        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                prev_activation = x
            else:
                prev_activation = self.layers[i - 1].activation

            # Calculate fun jacobians
            Diag_J_Z_Sum = self.layers[i].activation_function.df(
                self.layers[i].weighted_sums
            )
            J_Z_Y = np.einsum("ij,i->ij", self.layers[i].weights.T, Diag_J_Z_Sum)
            J_Z_W_Hat = np.outer(prev_activation, Diag_J_Z_Sum)
            J_L_W = J_L_Z * J_Z_W_Hat  # Weight
            J_L_B = J_L_Z * Diag_J_Z_Sum  # Bias
            J_L_Y = J_L_Z @ J_Z_Y

            # Update weights and bias
            self.layers[i].weights -= self.layers[
                i
            ].lr * J_L_W + self.alpha * self.regularization(self.layers[i].weights)
            self.layers[i].bias_weights -= self.layers[
                i
            ].lr * J_L_B + self.alpha * self.regularization(self.layers[i].bias_weights)

            # Loss for the next layer upstream is this
            J_L_Z = J_L_Y

    def train(self):
        train_losses = []
        val_losses = []
        epochs = np.linspace(1, self.epochs, self.epochs)
        initial_lr = []
        for l in self.layers:
            initial_lr.append(l.lr)

        for i in range(self.epochs):
            print("Epoch: ", i + 1)
            for x, y in zip(self.x_train, self.y_train):
                self.backward_pass(x, y)
            train_loss = 0
            for x, y in zip(self.x_train, self.y_train):
                train_loss += self.loss_function.f(y, self.forward_pass(x))
            train_losses.append(train_loss)

            # val_loss = 0
            # for x, y in zip(self.x_val, self.y_val):
            #     val_loss += self.loss_function.f(y, self.forward_pass(x))
            # val_losses.append(val_loss)
        accuracy = 0
        for x, y in zip(self.x_test, self.y_test):
            accuracy += np.argmax(self.forward_pass(x)) == np.argmax(self.y_test)
        print("Accuracy: ", accuracy / self.x_test.shape[0])
        ax = sns.lineplot(x=epochs, y=train_losses)
        #ax = sns.lineplot(x=epochs, y=val_losses, ax=ax)
        ax.set(xlabel="Epoch", ylabel="Error")
        #ax.legend([f"Train loss", f"Val loss"])
        plt.show()

