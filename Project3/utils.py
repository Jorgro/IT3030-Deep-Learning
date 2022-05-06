import matplotlib.pyplot as plt
import model_settings
import numpy as np
import keras
import random


def reshape_for_LSTM(X: np.array, k: int) -> np.array:
    """Reshapes data for training a LSTM model.

    Args:
        X (np.array): Data to reshape.
        k (int): Sequence length.

    Returns:
        np.array: Reshaped data.
    """
    n = X.shape[0]
    X_p = np.zeros((n - k, k, X.shape[1]))
    for i in range(k, n):
        X_p[i - k, :, :] = X[i - k : i, :].reshape(1, k, X_p.shape[2])
    return X_p


def plot_random(X: np.array, y: np.array, N: int, model, file_name=None):
    """Plots N forecasts from a model and saves it to file.

    Args:
        X (np.array): Input.
        y (np.array): Target.
        N (int): Number of graphs.
        model: Model to use for forecasting.
        file_name (str, optional): File to save graph. Defaults to None.
    """
    forecast_window_len = 24

    fig = plt.figure(figsize=(20, 20))
    columns = 5
    rows = N // columns
    seq_len = model_settings.SEQUENCE_LENGTH

    for i in range(1, columns * rows + 1):

        start_ind = random.randint(seq_len, X.shape[0] - seq_len - 1)
        forecasts = model.forecast(X, start_ind, forecast_window_len)

        fig.add_subplot(rows, columns, i)
        plt.plot(
            range(seq_len - 1, seq_len + forecast_window_len - 1),
            y[start_ind + seq_len - 1 : start_ind + seq_len + forecast_window_len - 1],
            label="y_true",
        )
        plt.plot(
            range(seq_len - 1, seq_len + forecast_window_len - 1),
            forecasts,
            label="y_pred",
        )
        plt.plot(range(0, seq_len), y[start_ind : start_ind + seq_len], label="hist")
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    plt.show()


class TrainingVisualizationCb(keras.callbacks.Callback):
    """Class used for visualizing training loss as a keras callback.
    """
    def __init__(self, file_name):
        self.file_name = file_name

    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses, label="Val loss")
        plt.legend(loc="upper left")
        plt.savefig(self.file_name, bbox_inches="tight")
        plt.show()
