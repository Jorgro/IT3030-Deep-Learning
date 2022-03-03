from Project2.gen_net import GenerativeNetwork
import numpy as np
from tensorflow import keras
import tensorflow as tf

class AutoEncoder(GenerativeNetwork):
    def __init__(self, file_name="./models/ae", latent_dim=8):
        self.bce = keras.losses.BinaryCrossentropy()

    def find_anomalies(self, x, k = 10):
        reconstructed = self.reconstruct(x)

        losses = []
        for i in range(x.shape[0]):
            losses.append(self.bce(x[i], reconstructed[i]).numpy())

        ind = np.argpartition(losses, -k)[-k:] # Get the indices of the k largest losses
        return ind