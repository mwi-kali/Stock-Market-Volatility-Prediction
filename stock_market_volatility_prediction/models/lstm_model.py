import tensorflow as tf


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import Tuple


class LSTMModel:
    def __init__(self, input_shape: Tuple[int,int], dropout: float = 0.3, l2_reg: float = 1e-3, lr: float = 1e-3):
        self.input_shape = input_shape
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.model = self._build()

    def _build(self) -> tf.keras.Model:
        tf.keras.backend.clear_session()
        m = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape, kernel_regularizer=l2(self.l2_reg)),
            Dropout(self.dropout),
            LSTM(64, kernel_regularizer=l2(self.l2_reg)),
            Dropout(self.dropout),
            Dense(1)
        ])
        m.compile(optimizer=Adam(learning_rate=self.lr), loss="mean_squared_error", metrics=["mae"])
        return m

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32, patience: int = 10):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save(path)

    @staticmethod
    def load(path: str):
        return tf.keras.models.load_model(path)
