import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from task_manager import STAGE, Task
from win_generator import WindowGenerator


class JobPredict:
    def __init__(self, task: Task) -> None:
        self._task = task

        self._train_task = Task(self._task.prev_task)
        assert self._train_task.stage == STAGE.TRAIN
        self._model = keras.models.load_model(self._train_task.keras_model)

        self._dataset_task = self._task.parent(match_func=lambda t: t.stage == STAGE.MAKE_DATASET)
        self._add_feat_task = self._task.parent(match_func=lambda t: t.stage == STAGE.ADD_FEATURE)

        self._pred_probas = {}

    def keras_eval(self, role):
        df = self._dataset_task.load_dataset(role)

        wg = WindowGenerator(
            win_size=self._train_task.config["seq_len"],
            all_cols=list(df.columns),
            feat_groups=self._train_task.config["feat_groups"],
            only_feature=False,  # evaluate also needs label and sample_weight
        )

        dataset = wg.make_dataset(df, batch_size=self._task.config.get("batch_size", 128), shuffle=False)
        return self._model.evaluate(dataset, return_dict=True)

    def predict(self, X):
        seq_len = self._train_task.config["seq_len"]
        wg = WindowGenerator(
            win_size=seq_len,
            all_cols=list(X.columns),
            feat_groups=self._train_task.config["feature_group"],
            only_feature=True,  # no label and sample_weight
        )
        dataset = wg.make_dataset(X, batch_size=self._task.config.get("batch_size", 128), shuffle=False)

        predictions = self._model.predict(dataset)  # [N,1]
        # first SEQ_LEN-1 examples cannot build valid window, hence are ignored during prediction
        # first label available at index=seq_len - 1, which is seq_len from start
        return pd.Series(predictions.flatten(), index=X.index[(seq_len - 1) :], name="pred_up_proba")

    def _dump4_vector_backtest(self):
        raw_features = self._add_feat_task.load_raw_features()  # feature 'return' not scaled yet
        returns = raw_features.loc[:, "return_b1d"]  # today - yesterday return
        for role, _probas in self._pred_probas.items():
            df = pd.concat([returns, _probas], axis=1, join="inner")

            fname = self._task.vector_bt_file(role, "in")
            df.to_csv(fname, index_label="Date")

    def _dump4_event_backtest(self):
        symbol = self._add_feat_task.config["symbol"]
        ohlc_data = Task.load_ohlcv(symbol)

        for role, _probas in self._pred_probas.items():
            bt_input = pd.concat([ohlc_data, _probas], axis=1, join="inner")
            bt_input.to_csv(self._task.bt_input(role), index_label="Date")

    def _find_pred_percentiles(self):
        for role, probas in self._pred_probas.items():
            sorted_scores = probas.sort_values()

            percentiles = {}
            for n in range(1, 10):
                index = int(n / 10 * len(sorted_scores))
                # original value in series is float32, which isn't JSON serializable
                percentiles[f"{n*10}p"] = float(sorted_scores[index])

            self._task.outputs[f"{role}_proba_percentiles"] = percentiles

    def main(self):
        aucs = {}
        score_stddevs = {}
        for role in ["train", "val", "test"]:
            df = self._dataset_task.load_dataset(role)

            _probs = self.predict(df)
            self._pred_probas[role] = _probs

            aucs[role] = roc_auc_score(y_true=df.loc[_probs.index, "forward_move"], y_score=_probs)
            score_stddevs[role] = np.std(_probs)  # larger stddev, larger discernibility

        self._task.outputs["AUC"] = aucs
        self._task.outputs["score_stddev"] = score_stddevs
        self._find_pred_percentiles()

        self._dump4_vector_backtest()
        self._dump4_event_backtest()

        return self._task.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task")
    args = parser.parse_args()

    task = Task(
        prev_task=args.prev_task if args.prev_task is not None else Task.latest_task().task_id,
        stage=STAGE.PREDICT,
    )
    JobPredict(task).main()
