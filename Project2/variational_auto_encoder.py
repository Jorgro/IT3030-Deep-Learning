from gen_net import GenerativeNetwork
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp


class VariationalAutoEncoder(GenerativeNetwork):
    def __init__(self, latent_dim=8, missing=False):
        GenerativeNetwork.__init__(self, "./models/vae", latent_dim, missing)
        self.bce = keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        standard_normal = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(self.latent_dim), scale=1.0),
            reinterpreted_batch_ndims=1,
        )
        input_img = keras.Input(shape=(28, 28, 1))
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
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(
            tfp.layers.IndependentNormal.params_size(self.latent_dim), activation=None,
        )(x)
        # self.latent_dim params for both mean and deviation -> Total 2*self.latent_dim

        encoded = tfp.layers.IndependentNormal(
            self.latent_dim,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                standard_normal, weight=0.5
            ),
        )(x)
        # Encoded layers as multivariate normal distribution of size self.latent_dim
        # Also adding loss (regularizer) between this distribution and the prior (standard normal)

        decoder_input = layers.InputLayer(input_shape=self.latent_dim)(encoded)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(7 * 7 * 32, activation="relu")(decoder_input)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")(
            x
        )
        x = layers.Flatten()(x)
        decoded = tfp.layers.IndependentBernoulli((28, 28, 1))(x)
        # Independent Bernoulli layer works by having the x as input params into a multivariate Bernoulli distribution
        # for the pixels which we can then sample, mean or mode from.
        self.model = keras.Model(input_img, decoded)
        opt = keras.optimizers.Adam(learning_rate=0.001)

        def neg_log_likelihood(x, y):
            return -y.log_prob(x)

        self.model.compile(
            optimizer=opt, loss=neg_log_likelihood
        )
        # Negative ELBO loss = w*KL_div + neg_log_likelihood

        self.decoder = keras.Model(decoder_input, decoded)
        self.encoder = keras.Model(input_img, encoded)

    def get_anomalies(self, x, k=10, N=10000):
        no_channels = x.shape[-1]
        decoded_z = np.zeros((N, 28, 28, no_channels))
        for i in range(no_channels):
            z = np.random.randn(N, self.latent_dim)
            decoded_z[:, :, :, [i]] = self.decoder(z).mode()

        losses = []
        for i in range(x.shape[0]):
            y = x[i]
            y = np.repeat(y[np.newaxis, :, :, :], N, axis=0)
            loss = self.bce(y, decoded_z).numpy()
            loss = np.average(loss, axis=(1, 2))
            loss = np.exp(-loss)
            loss = np.sum(loss) / N
            losses.append(loss)
        ind = np.argpartition(
            losses, k
        )  # Get the indices of the k lowest probabilities
        return ind

