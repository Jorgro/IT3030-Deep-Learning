from gen_net import GenerativeNetwork
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class AutoEncoder(GenerativeNetwork):
    def __init__(self, latent_dim=8, missing=False):
        GenerativeNetwork.__init__(self, "./models/ae", latent_dim, missing)
        self.bce = keras.losses.BinaryCrossentropy()

        input_img = keras.Input(shape=(28, 28, 1))

        # Encoder
        x = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="valid",
            activation="relu",
        )(input_img)
        x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="valid",
            activation="relu",
        )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        encoded = layers.Dense(self.latent_dim)(x)

        # Decoder
        decoder_input = layers.InputLayer(input_shape=self.latent_dim)(encoded)
        x = layers.Dense(128, activation="relu")(decoder_input)
        x = layers.Dense(7 * 7 * 32, activation="relu")(x)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding="same", activation="relu"
        )(x)
        decoded = layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding="same", activation="sigmoid"
        )(x)

        self.model = keras.Model(input_img, decoded)
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss="binary_crossentropy")

        self.decoder = keras.Model(decoder_input, decoded)
        self.encoder = keras.Model(input_img, encoded)

    def get_anomalies(self, x, k=10):
        reconstructed = self.reconstruct(x)

        losses = []
        for i in range(x.shape[0]):
            losses.append(self.bce(x[i], reconstructed[i]).numpy())

        ind = np.argpartition(losses, -k)[
            -k:
        ]  # Get the indices of the k largest losses
        return ind
