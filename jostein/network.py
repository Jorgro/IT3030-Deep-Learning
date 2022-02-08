import numpy as np
from layer import Layer
from activation_functions_j import Softmax
from loss_functions import MSE, CrossEntropy
import json
import pickle

"""
The network class keeps track of the layers, and calculates the loss.
The class implements fit() and predict() methods, as is conventionally the names used for these methods in ML.
"""


class Network:
    def __init__(self, networkConfig) -> None:
        self.layers = []
        # Create layers
        input_dim = networkConfig["input_dimension"]
        for c in networkConfig["layers"]:
            self.layers.append(
                Layer(
                    input_dim,
                    c["size"],
                    c["activation_function"],
                    networkConfig["regularization_weight"],
                    c["learning_rate"],
                )
            )
            input_dim = c["size"]
        self.l1_alpha = networkConfig["regularization"]  # l1 regularization constant
        self.l2_alpha = 0  # l2 regularization constant
        self.loss_function = CrossEntropy  # loss function
        # Boolean value telling the network if Softmax should be applied to the last layer
        self.softmax = networkConfig["type"] == "Softmax"
        self.softmax_func = Softmax()

    # The fit method trains the network on the training set. Validation set can be added optionally.
    # This implementation does not support minibatching, performs the backward pass on one case at the time.
    def fit(self, x_train, y_train, epochs=20, valid=None, verbose=False):
        self.train_scores = []
        self.valid_scores = []
        # self.omega1 = []
        # self.omega2 = []
        for i in range(epochs):
            print("Epoch: ", i)
            score = 0
            for x, y in zip(x_train, y_train):
                # print("y_train: ", y)
                # print(x)
                result = self.forward_pass(x)
                # print("res: ", result)

                Jlz = self.loss_function.f_prime(y, result)
                # print("Jlz: ", Jlz)

                # Apply softmax if enabled
                if self.softmax:
                    Jsz = self.softmax_func.df(result)
                    Jlz = Jlz @ Jsz

                # Calculate loss
                loss = self.loss_function.f(y, result)

                score += loss

                # Perform the backward pass through the layers
                self.backward_pass(Jlz)

                # Update the weights in the layers
                self.update_weights()

                if verbose:
                    print(f"Input: {x} \nOutput: {result} \nTarget: {y} \nLoss: {loss}")

            # Save loss score
            self.train_scores.append(score / len(x_train))

            if valid:
                score = 0
                for x, y in zip(valid.x, valid.y):
                    score += self.loss_function.f(y, self.predict(x))

                # Save validation score
                self.valid_scores.append(np.array([i, score / (len(valid.x))]))

            o1 = 0
            o2 = 0
            # Calculate regularization costs
            for l in self.layers:
                o1 += np.sum(np.abs(l.W))
                o2 += 0.5 * np.sum(l.W ** 2)

            # Save regularization costs
            # self.omega1.append(np.array([i, o1 * self.l1_alpha]))
            # self.omega2.append(np.array([i, o2 * self.l2_alpha]))

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

    def backward_pass(self, Jlz):
        jlz = Jlz
        # print(jlz)
        for i in range(len(self.layers) - 1, -1, -1):
            # Go backwards through the network and update weights
            jlz = self.layers[i].backward_pass(jlz)

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
    with open("dataset-j.p", "rb") as file:
        data = pickle.load(file)
        x_train, y_train = (
            np.array(data["x_train"]),
            np.array(data["y_train"]),
        )
        x_test, y_test = (
            np.array(data["x_test"]),
            np.array(data["y_test"]),
        )

    # shuffler = np.random.permutation(len(x_train))
    # x_train = x_train[shuffler]
    # y_train = y_train[shuffler]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    print("Loading config.")
    with open("jostein/config_3_2.json") as f:
        config = json.load(f)
    nn = Network(config)
    print("Loading dataset")
    x_train, y_train, x_test, y_test = load_data()
    print("Training network")
    scores = nn.fit(x_train, y_train)
    print(scores)
    print(nn.test_loss(x_test, y_test))

