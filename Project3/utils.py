import matplotlib.pyplot as plt
import settings
import numpy as np
import keras


def plot_random(X, y, N, model):
    forecast_window_len = 24

    fig = plt.figure(figsize=(20, 20))
    columns = 5
    rows = N // columns
    seq_len = settings.SEQUENCE_LENGTH

    for i in range(1, columns * rows + 1):

        start_ind = np.random.choice(X.shape[0] - seq_len)
        model_input = X[start_ind : start_ind + seq_len]
        forecasts = []

        for pred_no in range(1, forecast_window_len + 1):
            forecast = model(np.array([model_input]))
            forecasts.append(forecast[0][0])
            model_input = X[start_ind + pred_no : start_ind + pred_no + seq_len]
            model_input[-pred_no:, -1] = np.array(forecasts)

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

    plt.show()


class TrainingVisualizationCb(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses, label="Val loss")
        plt.show()
