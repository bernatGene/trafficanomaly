import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LstmModel:
    def __init__(self, max_len, pred_len):
        self.model = None
        self.max_len = max_len
        self.pred_len = pred_len
        self.get_model()
        self.load_model()

    def get_model(self):
        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(self.max_len, 2)),
                layers.LSTM(64),
                layers.Dense(128, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(2 * self.pred_len, activation=None),  # Linear activation
            ]
        )
        self.model = model

    def load_model(self):
        self.model.load_weights('trained_colab2_83.h5')

    @tf.function
    def _serve(self, x):
        return self.model(x, training=False)

    def predict(self, points):
        points = np.array(points)
        predict = self._serve(np.expand_dims(points,0))
        return predict.numpy().reshape(self.pred_len, 2)
