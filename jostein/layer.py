import numpy as np
from activation_functions_j import Sigmoid, ReLU, Linear, Tanh


class Layer:
    def __init__(self, input_nodes, output_nodes, activation, alpha, lr) -> None:
        # init random weights between 0 and 1
        initial_weight_range = (-0.5, 0.5)
        W = np.random.rand(input_nodes, output_nodes)
        # self.W = W
        # Scale initial weights
        self.W = W * initial_weight_range[0] + (1 - W) * initial_weight_range[1]
        # init random bias weights between 0 and 1
        b = np.random.rand(output_nodes)
        # Scale bias weights
        self.b = b * initial_weight_range[0] + (1 - b) * initial_weight_range[1]
        self.lr = lr
        self.l1_alpha = alpha
        self.l2_alpha = 0  # layerConfig.l2_alpha
        self.a = None  # summed outputs
        self.z = None  # f(a)

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
        self.a = y @ self.W + self.b
        self.z = self.activation.f(self.a)

        return self.z.copy()

    # Jlz is the jacobian matrix J L/Z , where L is the loss function and Z is the next layer
    def backward_pass(self, Jlz):
        self.df = self.activation.df(self.a)  # df = JZSum-diagonal
        self.Jzy = np.einsum("ij,i->ij", self.W.T, self.df)  # Numerator form
        self.Jzw_hat = np.outer(self.y, self.df)
        # print("J_Z_Y: ", self.Jzy.shape)
        # print("J_L_Z: ", Jlz.shape)
        # print("Jzw_hat: ", self.Jzw_hat.shape)

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
