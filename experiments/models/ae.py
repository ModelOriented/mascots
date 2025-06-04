from typing import Any

import tensorflow as tf
from tensorflow import keras


# from glacier
def Autoencoder(n_timesteps: int, n_features: int) -> Any:
    # Define encoder and decoder structure
    def Encoder(input: tf.Tensor) -> Any:
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )(input)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )(x)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        return x

    def Decoder(input: tf.Tensor) -> Any:
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )(input)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )(x)
        # x = keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=1, kernel_size=3, activation="linear", padding="same"
        )(x)
        return x

    # Define the AE model
    orig_input = keras.Input(shape=(n_timesteps, n_features))
    autoencoder = keras.Model(
        inputs=orig_input, outputs=Decoder(Encoder(orig_input))
    )

    return autoencoder
