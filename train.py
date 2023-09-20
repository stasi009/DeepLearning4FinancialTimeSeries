import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from win_generator import WindowGenerator
from task_manager import STAGE, Task


def find_best_fit_metrics(fit_history):
    best_metrics = {}
    for metric_name, metric_vals in fit_history.history.items():
        if metric_name.endswith("loss"):
            best_epoch = np.argmin(metric_vals)
        elif metric_name.endswith("auc"):
            best_epoch = np.argmax(metric_vals)
        else:
            raise ValueError(f"unknown metric name={metric_name}")
        best_metrics["best " + metric_name] = f"{metric_vals[best_epoch]:.4f} @ epoch={best_epoch+1}"
    return best_metrics


class JobTrain:
    def __init__(self, model, task: Task) -> None:
        self._task = task

        self._prev_task = Task(task.prev_task)
        assert self._prev_task.stage == STAGE.MAKE_DATASET

        self._configs = self._task.config
        self._model = model

    def load_datas(self):
        train_df = self._prev_task.load_dataset("train")

        self._win_gen = WindowGenerator(
            win_size=self._configs["seq_len"],
            all_cols=list(train_df.columns),
            feat_groups=self._configs["feature_group"],
            only_feature=False,  # features + label + sample_weight
        )

        batch_size = self._configs.get("batch_size", 128)
        # if shuffle=True indicating shuffle the window within or across batches,
        # but inside one window, datas (i.e., rows) are still chronologically ordered
        self.train_dataset = self._win_gen.make_dataset(train_df, batch_size=batch_size, shuffle=True)

        val_df = self._prev_task.load_dataset("val")
        self.val_dataset = self._win_gen.make_dataset(val_df, batch_size=batch_size, shuffle=False)

    def fit(self):
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._configs.get("learning_rate", 1e-3)),
            loss=keras.losses.BinaryCrossentropy(),
            # must name the metric, otherwise if compile two different models in one process,
            # the second AUC metric will be auto-named as "auc_1"
            weighted_metrics=[keras.metrics.AUC(name="auc")],
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self._task.keras_model,
                save_best_only=True,
                monitor="val_auc",
                mode="max",
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self._configs.get("earlystop_patience", 3),
            ),
            keras.callbacks.TensorBoard(
                self._task.keras_log, update_freq=self._configs.get("log_freq", 10)
            ),
        ]

        fit_history = self._model.fit(
            self.train_dataset,
            epochs=self._configs.get("epochs", 10),
            validation_data=self.val_dataset,
            callbacks=callbacks,
        )
        self._task.outputs["fit_metrics"] = find_best_fit_metrics(fit_history)

    def refit(self):
        """refit on validation dataset"""
        model = keras.models.load_model(self._task.keras_model)

        val_df = self._prev_task.load_dataset("val")
        dataset = self._win_gen.make_dataset(
            val_df, batch_size=self._configs.get("batch_size", 128), shuffle=False
        )

        history = model.fit(
            dataset,
            epochs=1,  # just train one epoch, avoid overfitting
        )
        model.save(self._task.refitted_keras_model)
        self._task.outputs["refit_metrics"] = find_best_fit_metrics(history)

    def main(self, set_seed=True, refit=False):
        if set_seed:
            rand_seed = self._task.config["rand_seed"]
            if rand_seed is None:
                rand_seed = random.randint(1, 90000)
                self._task.config["rand_seed"] = rand_seed
            tf.random.set_seed(rand_seed)
            tf.keras.utils.set_random_seed(rand_seed)

        self.load_datas()
        self.fit()
        if refit:
            self.refit()

        return self._task.done()
