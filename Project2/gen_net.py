import abc
import numpy as np
from stacked_mnist import DataMode, StackedMNISTData
import tensorflow_probability as tfp


class GenerativeNetwork(abc.ABC):
    def __init__(self, file_name, latent_dim=8, missing=False):
        self.missing = missing
        self.latent_dim = latent_dim
        if self.missing:
            self.file_name = f"{file_name}_missing"
        else:
            self.file_name = file_name

    def train(self, epochs: np.int = 10, force_relearn=False):
        print(self.file_name)
        print(self.missing)
        if not force_relearn:
            self.done_training = self.load_weights()
        if force_relearn or not self.done_training:
            if self.missing:
                generator = StackedMNISTData(
                    mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048
                )
            else:
                generator = StackedMNISTData(
                    mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048
                )
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            self.model.fit(
                x_train,
                x_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
            )

            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

    def load_weights(self):
        try:
            self.model.load_weights(filepath=self.file_name)
            print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights from file. Must retrain...")
            done_training = False

        return done_training

    def generate_new_samples(self, no_channels=1, no_new_samples=1000):
        generated = np.zeros((no_new_samples, 28, 28, no_channels))
        for i in range(no_channels):
            z = np.random.randn(no_new_samples, self.latent_dim) * 100
            y = self.decoder(z)
            # Yes, very ugly check if output layer is of type probability -> VAE
            if isinstance(y, tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible):
                y = y.mode()
            generated[:, :, :, [i]] = y.numpy()
        return generated

    def reconstruct(self, x):
        no_channels = x.shape[-1]
        reconstructed = np.zeros(x.shape)
        for i in range(no_channels):
            y = self.model(x[:, :, :, [i]])
            if isinstance(y, tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible):
                y = y.mode()
            reconstructed[:, :, :, [i]] = y.numpy()
        return reconstructed

