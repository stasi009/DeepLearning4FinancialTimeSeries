import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    def __init__(self, configs) -> None:
        self.configs = configs

    @abstractmethod
    def build(self):
        pass


class SimpleRNN(BaseModel):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self._feat_names = configs["feature_group"]["feature"]["feat_names"]

    def build(self):
        l2 = self.configs.get("l2", 0)

        model = keras.Sequential()
        model.add(keras.Input(shape=(self.configs["seq_len"], len(self._feat_names))))

        for idx, units in enumerate(self.configs["rnn"]):
            return_sequences = idx != len(self.configs["rnn"]) - 1
            lstm = layers.LSTM(
                name=f"lstm_{idx}",
                units=units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(l2),
                recurrent_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l2(l2),
            )
            model.add(lstm)

        for idx, units in enumerate(self.configs["dense"]):
            hidden = layers.Dense(
                name=f"hidden_{idx}",
                units=units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2),
            )
            model.add(hidden)

        final_layer = layers.Dense(name="final", units=1, activation="sigmoid")
        model.add(final_layer)

        return model


class ImproveRnnWithIndicators(BaseModel):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self._realnum_features = configs["feature_group"]["realnum"]["feat_names"]
        self._indicator_features = configs["feature_group"]["indicator"]["feat_names"]

    def build(self):
        l2 = self.configs.get("l2", 0)

        # history: [batch_size, window_size, feature_dim]
        realnum_input = keras.Input(
            name="realnum", shape=(self.configs["seq_len"], len(self._realnum_features))
        )
        # current step: [batch_size, feature_dim]
        indicator_input = keras.Input(name="indicator", shape=(len(self._indicator_features),))

        X = realnum_input
        for idx, units in enumerate(self.configs["rnn"]):
            return_sequences = idx != len(self.configs["rnn"]) - 1
            lstm = layers.LSTM(
                name=f"lstm_{idx}",
                units=units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(l2),
                recurrent_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l2(l2),
            )
            # input: [batch_size, window_size, units]
            # output: [batch_size, window_size, units] or [batch_size, units]
            X = lstm(X)

        # [batch_size, ind_feature_dim + rnn_units]
        X = tf.concat([indicator_input, X], axis=1)

        for idx, units in enumerate(self.configs["dense"]):
            dense = layers.Dense(
                name=f"dense_{idx}",
                units=units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2),
            )
            X = dense(X)

        final_layer = layers.Dense(name="final", units=1, activation="sigmoid")
        logit = final_layer(X)

        model = keras.Model(inputs=[realnum_input, indicator_input], outputs=logit)
        model.summary()

        return model
