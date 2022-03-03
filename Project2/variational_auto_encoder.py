from Project2.gen_net import GenerativeNetwork
import numpy as np
from tensorflow import keras
import tensorflow as tf


def VariationalAutoEncoder(GenerativeNetwork):
    def __init__(self, file_name="./models/vae", latent_dim=8):
        self.bce = keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def find_anomalies(self, x, k=10, N=10000):
        no_channels = x.shape[-1]
        decoded_z = np.zeros((N, 28, 28, no_channels))
        for i in range(3):
            z = np.random.randn(N, 8)
            decoded_z[:, :, :, [i]] = self.decoder(z).mode()

        losses = []
        for i in range(x.shape[0]):
            x = x[i]
            x = np.repeat(x[np.newaxis, :, :, :], N, axis=0)
            loss = self.bce(x, decoded_z).numpy()
            loss = np.average(loss, axis=(1, 2))
            loss = np.exp(loss)
            loss = np.sum(loss) / N
            losses.append(loss)

        ind = np.argpartition(losses, -k)[
            -k:
        ]  # Get the indices of the k largest losses
        return ind

