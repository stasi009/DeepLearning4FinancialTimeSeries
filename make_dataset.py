from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from task_manager import Task, STAGE
import argparse


class FeatProcMethod(Enum):
    COPY = 1
    SCALING = 2
    LOG_SCALING = 3
    USER_DEFINE = 4


class JobMakeDataset:
    def __init__(self, task: Task) -> None:
        self.df = None
        self.task = task
        self.prev_task = Task(task_id=self.task.prev_task)
        assert self.prev_task.stage == STAGE.ADD_FEATURE

        splits = task.config["splits"]
        assert len(splits) == 3
        self.splits = [x / sum(splits) for x in splits]

        # key is feature name, value = process method
        self.keep_features = self.task.config["keep_features"]
        self.keep_features["forward_move"] = FeatProcMethod.COPY  # always need to copy label
        self.keep_features["forward_change"] = FeatProcMethod.USER_DEFINE

        # ---------------- feature scaling
        self._scaling_cols = [
            c
            for c, m in self.keep_features.items()
            if m == FeatProcMethod.SCALING or m == FeatProcMethod.LOG_SCALING
        ]
        self._feat_scaler = StandardScaler()

        # ---------------- sample weight scaling
        if self.task.config["max_sample_weight"] > 1:
            assert self.task.config["sample_weight_bins"] > 1
            self._weight_scaler = KBinsDiscretizer(
                n_bins=self.task.config["sample_weight_bins"],
                encode="ordinal",
                strategy="uniform",
            )
        else:
            self._weight_scaler = None

    def __split(self):
        total_examples = self.df.shape[0]
        train_size = int(total_examples * self.splits[0])
        val_size = int(total_examples * self.splits[1])

        train_df = self.df.iloc[:train_size, :]
        val_df = self.df.iloc[train_size : (train_size + val_size), :]
        test_df = self.df.iloc[(train_size + val_size) :, :]

        return train_df, val_df, test_df

    def __scale_dump(self, old_df, role):
        scaled = self._feat_scaler.transform(old_df.loc[:, self._scaling_cols])  # return ndarray
        new_df = pd.DataFrame(scaled, index=old_df.index, columns=self._scaling_cols)

        for col, method in self.keep_features.items():
            if method == FeatProcMethod.COPY:
                new_df[col] = old_df[col]

        if self._weight_scaler is None:
            new_df["sample_weight"] = 1
        else:
            # weight_bins range from 0 to n_bins-1
            weight_bins = self._weight_scaler.transform(old_df.loc[:, ["forward_change"]])
            new_df["sample_weight"] = 1 + (self.task.config["max_sample_weight"] - 1) * weight_bins / (
                self.task.config["sample_weight_bins"] - 1
            )

        self.task.save_dataset(role=role, df=new_df)

        # fill result
        start_day, end_day = (new_df.index[idx].to_pydatetime().strftime("%Y%m%d") for idx in [0, -1])
        weight_counts = new_df["sample_weight"].value_counts().to_dict()
        self.task.outputs[f"{role} dataset"] = {
            "size": f"'{start_day}'~'{end_day}' shape={new_df.shape}",
            "positive-rate": new_df["forward_move"].mean(),
            "sample_weights": [
                f"sample_weight={weight}, count={count:6d}, ratio={count*100 / new_df.shape[0]:>8.3f}%"
                for weight, count in weight_counts.items()
            ],
        }

    def __fit(self, train_df):
        self._feat_scaler.fit(train_df.loc[:, self._scaling_cols])

        if self._weight_scaler is not None:
            self._weight_scaler.fit(train_df.loc[:, ["forward_change"]])

    def main(self):
        self.df = self.prev_task.load_raw_features()

        # --------------- only keep important features
        self.df = self.df.loc[:, list(self.keep_features.keys())]

        # --------------- transform
        self.df["forward_change"] = self.df["forward_change"].abs()

        for col, method in self.keep_features.items():
            if method == FeatProcMethod.LOG_SCALING:
                self.df[col] = np.log1p(self.df[col])

        # --------------- split
        train_df, val_df, test_df = self.__split()

        # --------------- scaling
        self.__fit(train_df)
        self.__scale_dump(train_df, "train")
        self.__scale_dump(val_df, "val")
        self.__scale_dump(test_df, "test")

        self.task.config["keep_features"] = {
            c: m.name for c, m in self.keep_features.items()
        }  # JSON compatible
        return self.task.done()


def keep_features():
    # **************** real-number columns
    feat_names = [
        "Close",
        "Volume",
        "c-o",
        "h-l",
        "macdsignal",
        "macdhist",
        "chaikin",
        "obv",
        "return_b1d",
        "return_b5d",
        "return_b5d_std",
        "return_b10d",
        "return_b10d_std",
        "return_b20d",
        "return_b20d_std",
        "vlm_5d_avg",
        "vlm_5d_std",
        "vlm_10d_avg",
        "vlm_10d_std",
        "vlm_20d_avg",
        "vlm_20d_std",
        "price_5d_sma",
        "price_10d_sma",
        "price_20d_sma",
        "price_5d_std",
        "price_10d_std",
        "price_20d_std",
        "rsi_5d",
        "rsi_10d",
        "rsi_20d",
        "atr_5d",
        "atr_10d",
        "atr_20d",
    ]
    feat_proc_methods = {c: FeatProcMethod.LOG_SCALING for c in ["Close", "Volume", "h-l"]}
    feat_proc_methods["c-o"] = FeatProcMethod.SCALING
    for c in ["chaikin", "obv"]:
        feat_proc_methods[c] = FeatProcMethod.SCALING

    for c in feat_names:
        if c.startswith("atr_"):
            feat_proc_methods[c] = FeatProcMethod.LOG_SCALING

        if c.startswith("price_"):
            feat_proc_methods[c] = FeatProcMethod.LOG_SCALING

        if c.startswith("return"):
            feat_proc_methods[c] = FeatProcMethod.SCALING

        if c.startswith("vlm_"):
            feat_proc_methods[c] = FeatProcMethod.LOG_SCALING

        if c.startswith("rsi_"):
            feat_proc_methods[c] = FeatProcMethod.SCALING

        if c.startswith("macd"):
            feat_proc_methods[c] = FeatProcMethod.SCALING

    if len(feat_proc_methods) != len(feat_names):
        raise Exception("some feature missing process methods")

    # **************** indicator columns
    for c in [
        "is_macd_gold",
        "is_macd_dead",
        "is_pricema_gold",
        "is_pricema_dead",
        "is_rsi5_gold",
        "is_rsi5_dead",
        "is_boll10_gold",
        "is_boll10_dead",
        "is_boll20_gold",
        "is_boll20_dead",
    ]:
        feat_proc_methods[c] = FeatProcMethod.COPY

    return feat_proc_methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task")
    parser.add_argument("-wb", "--sample_weight_bins", type=int, default=-1)
    parser.add_argument("-mw", "--max_sample_weight", type=float, default=1)
    args = parser.parse_args()

    task = Task(
        prev_task=args.prev_task if args.prev_task is not None else Task.latest_task().task_id,
        stage=STAGE.MAKE_DATASET,
        config={
            "keep_features": keep_features(),
            "splits": [2, 1, 1],
            "sample_weight_bins": args.sample_weight_bins,
            "max_sample_weight": args.max_sample_weight,
        },
    )
    JobMakeDataset(task).main()
