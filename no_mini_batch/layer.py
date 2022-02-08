import numpy as np
from activation_functions_j import Sigmoid, ReLU, Linear, Tanh


class Layer:
    def __init__(
        self, input_nodes, output_nodes, activation, lr, weights, bias_weights
    ) -> None:

        # init random weights between 0 and 1
        initial_weight_range = (-0.5, 0.5)
        W = np.random.rand(input_nodes, output_nodes)
        self.W = W
        # Scale initial weights
        self.W = W * initial_weight_range[0] + (1 - W) * initial_weight_range[1]
        # init random bias weights between 0 and 1
        b = np.random.rand(output_nodes)
        # Scale bias weights
        self.b = b * initial_weight_range[0] + (1 - b) * initial_weight_range[1]
        self.W = weights
        self.b = bias_weights

        self.lr = lr
        self.z = None  # f(a)
        self.J_L_Z = None

        if activation == "Sigmoid":
            self.activation = Sigmoid()
        elif activation == "ReLU":
            self.activation = ReLU()
        elif activation == "Linear":
            self.activation = Linear()
        else:
            activation = Tanh()

    # Performs the forward pass given input y (row vector)
    def forward_pass(self, y) -> np.array:

        # print(y.shape)
        # print(self.W.shape)
        self.y = y
        self.weighted_sums = y @ self.W + self.b
        self.z = self.activation.f(self.weighted_sums)

        return self.z.copy()

    # Jlz is the jacobian matrix J L/Z , where L is the loss function and Z is the next layer
    def backward_pass(self, Jlz):
        self.df = self.activation.df(self.weighted_sums)  # df = JZSum-diagonal
        self.Jzy = np.einsum("ij,i->ij", self.W.T, self.df)  # Numerator form
        self.Jzw_hat = np.outer(self.y, self.df)
        self.Jly = Jlz @ self.Jzy
        self.Jlw = Jlz * self.Jzw_hat  # Weight derivative
        self.Jlb = Jlz * self.df  # Bias derivative

        return self.Jly.copy()

    # Update weights and biases
    def update_weights(self):
        self.W -= (
            self.lr * self.Jlw
            + self.l1_alpha * np.sign(self.W)
            + self.l2_alpha * self.W
        )

        self.b -= self.lr * self.Jlb

    def __str__(self) -> str:
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}"

    def __repr__(self):
        return f"Layer - shape: {self.W.shape}, f(a): {self.z}, activation: {self.activation}"
