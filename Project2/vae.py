from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfd
import tensorflow as tf


class VAE_MNIST:
    def __init__(self, dim_z, kl_weight, learning_rate):
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

    # Sequential API encoder
    def encoder_z(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfd.distributions.Independent(
            tfd.distributions.Normal(loc=tf.zeros(self.dim_z), scale=1.0),
            reinterpreted_batch_ndims=1,
        )
        # build layers argument for tfk.Sequential()
        input_shape = self.dim_x
        model_layers = [keras.layers.InputLayer(input_shape=input_shape)]
        model_layers.append(
            layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="valid",
                activation="relu",
            )
        )
        model_layers.append(
            layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="valid",
                activation="relu",
            )
        )
        model_layers.append(layers.Flatten())
        # the following two lines set the output to be a probabilistic distribution
        model_layers.append(
            layers.Dense(
                tfd.layers.IndependentNormal.params_size(self.dim_z),
                activation=None,
                name="z_params",
            )
        )
        model_layers.append(
            tfd.layers.IndependentNormal(
                self.dim_z,
                activity_regularizer=tfd.layers.KLDivergenceRegularizer(
                    prior, weight=self.kl_weight
                ),
                name="z_layer",
            )
        )
        return keras.Sequential(model_layers, name="encoder")

    # Sequential API decoder
    def decoder_x(self):
        model_layers = [layers.InputLayer(input_shape=self.dim_z)]
        model_layers.append(layers.Dense(7 * 7 * 32, activation=None))
        model_layers.append(layers.Reshape((7, 7, 32)))
        model_layers.append(
            layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding="same", activation="relu"
            )
        )
        model_layers.append(
            layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding="same", activation="relu"
            )
        )
        model_layers.append(
            layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")
        )
        model_layers.append(layers.Flatten(name="x_params"))
        # note that here we don't need
        # `tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.dim_x))` because
        # we've restored the desired input shape with the last Conv2DTranspose layer
        model_layers.append(tfd.layers.IndependentBernoulli(self.dim_x, name="x_layer"))
        return keras.Sequential(model_layers, name="decoder")

    def build_vae_keras_model(self):
        x_input = keras.Input(shape=self.dim_x)
        self.encoder = self.encoder_z()
        self.decoder = self.decoder_x()
        z = self.encoder(x_input)

        # compile VAE model
        model = keras.Model(inputs=x_input, outputs=self.decoder(z))
        model.compile(
            loss=lambda x, y: -y.log_prob(x),
            optimizer=keras.optimizers.Adam(self.learning_rate),
        )
        return model
