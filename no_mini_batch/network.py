import numpy as np
from layer import Layer
from activation_functions_j import Softmax
from loss_functions import MSE, CEE
import json
import pickle

"""
The network class keeps track of the layers, and calculates the loss.
The class implements fit() and predict() methods, as is conventionally the names used for these methods in ML.
"""


class NeuralNetwork:
    def __init__(self, networkConfig) -> None:
        self.layers = []
        # Create layers
        input_dim = networkConfig["input_dimension"]
        for layer in networkConfig["layers"]:
            if "learning_rate" in layer.keys():
                lr = layer["learning_rate"]
            else:
                lr = networkConfig["learning_rate"]

            weights = NeuralNetwork.get_range(
                layer["weight_range"], input_dim, layer["size"]
            )
            bias_weights = NeuralNetwork.get_range(
                layer["bias_range"], 1, layer["size"], one_dim=True
            )

            self.layers.append(
                Layer(
                    input_dim,
                    layer["size"],
                    layer["activation_function"],
                    lr,
                    weights,
                    bias_weights
                )
            )
            input_dim = layer["size"]
        self.alpha_method =  networkConfig["regularization"]
        self.alpha = networkConfig["regularization_weight"]
        if networkConfig["loss_function"] == "CEE":
            self.loss_function = CEE()
        else:
            self.loss_function = MSE()

        self.softmax = networkConfig["type"] == "Softmax"
        self.softmax_func = Softmax()

    @staticmethod
    def get_range(method, in_dim, out_dim, one_dim=False) -> np.ndarray:
        if method == "glorot":
            mean = 0
            variance = 2.0 / (in_dim + out_dim)
            v = np.random.normal(mean, variance, (in_dim, out_dim))
        elif isinstance(method, list):
            v = np.random.uniform(method[0], method[1], (in_dim, out_dim))
        else:
            v = np.random.uniform(-0.5, 0.5, (in_dim, out_dim))
        if one_dim:
            return v.flatten()
        return v

    # The fit method trains the network on the training set. Validation set can be added optionally.
    # This implementation does not support minibatching, performs the backward pass on one case at the time.
    def train(self, x_train, y_train, epochs=20):
        self.train_scores = []
        self.valid_scores = []
        for i in range(epochs):
            print("Epoch: ", i)
            score = 0
            for x, y in zip(x_train, y_train):
                # print("y_train: ", y)
                # print(x)
                y_pred = self.forward_pass(x)
                loss = self.loss_function.f(y, y_pred)

                # Calculate loss

                score += loss

                # Perform the backward pass through the layers
                self.backward_pass(x, y, y_pred)

                # Update the weights in the layers
                # self.update_weights()

                # if verbose:
                #    print(f"Input: {x} \nOutput: {result} \nTarget: {y} \nLoss: {loss}")

            # Save loss score
            self.train_scores.append(score / len(x_train))

            # if valid:
            #    score = 0
            #    for x, y in zip(valid.x, valid.y):
            #        score += self.loss_function.f(y, self.predict(x))#
        #
        # Save validation score
        #                self.valid_scores.append(np.array([i, score / (len(valid.x))]))

        return self.train_scores, self.valid_scores

    # Make a prediction
    def predict(self, x_test):
        return self.forward_pass(x_test)

    # Perform forward pass
    def forward_pass(self, x):
        o = x
        for l in self.layers:
            o = l.forward_pass(o)

        if self.softmax:
            return self.softmax_func.f(o)
        return o

    def update_weights(self):
        for l in self.layers:
            l.update_weights()

    def backward_pass(self, x, y, y_pred):
        J_L_Z = self.loss_function.df(y, y_pred)
        if self.softmax:
            Jsz = self.softmax_func.df(y_pred)
            J_L_Z = J_L_Z @ Jsz

        # print(jlz)
        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                prev_activation = x
            else:
                prev_activation = self.layers[i - 1].z

            Diag_J_Sum = self.layers[i].activation.df(self.layers[i].weighted_sums)
            J_Z_Y = np.einsum(
                "ij,i->ij", self.layers[i].W.T, Diag_J_Sum
            )  # Numerator form
            Jzw_hat = np.outer(prev_activation, Diag_J_Sum)
            J_L_W = J_L_Z * Jzw_hat  # Weight derivative
            J_L_B = J_L_Z * Diag_J_Sum  # Bias derivative
            J_L_Z = J_L_Z @ J_Z_Y

            self.layers[i].W -= (
                self.layers[i].lr * J_L_W
                + self.alpha * np.sign(self.layers[i].W)
                #+ self.layers[i].l2_alpha * self.layers[i].W
            )

            self.layers[i].b -= self.layers[i].lr * J_L_B

            # J_L_Z = self.layers[i].J_L_Z.copy()

            # Go backwards through the network and update weights
            # J_L_Z = self.layers[i].backward_pass(J_L_Z)

    # Calculate test loss from test_set
    def test_loss(self, x_test, y_test):
        score = 0
        accuracy = 0
        for x, y in zip(x_test, y_test):
            score += self.loss_function.f(y, self.predict(x))
            accuracy += np.argmax(y) == np.argmax(self.predict(x))

        score = score / x_test.shape[0]

        return accuracy / x_test.shape[0]

    def __str__(self) -> str:
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"

    def __repr__(self):
        return f"{self.layers} Softmax: {self.softmax} L1: {self.l1_alpha}, L2: {self.l2_alpha}"


def load_data():
    """
        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.
        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
    with open("dataset.pickle", "rb") as file:
        data = pickle.load(file)
        x_train, y_train = (
            np.array(data["x_train"]),
            np.array(data["y_train"]),
        )
        x_test, y_test = (
            np.array(data["x_test"]),
            np.array(data["y_test"]),
        )

    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    print("Loading config.")
    with open("no_mini_batch/config_1.json") as f:
        config = json.load(f)
    nn = NeuralNetwork(config)
    print("Loading dataset")
    x_train, y_train, x_test, y_test = load_data()
    print("Training network")
    scores = nn.train(x_train, y_train)
    print(scores)
    print(nn.test_loss(x_test, y_test))

