import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from train import JobTrain
import tensorflow as tf
from tensorflow import keras
from glob import glob
from task_manager import Task, STAGE, TASK_MANAGER
from win_generator import WindowGenerator
from bisect import bisect_left
from collections import defaultdict


def combine_backtest_results(symbol, role):
    files = glob(f"backtests/{symbol}_{role}_*_result.csv")
    bt_results = []
    for fname in files:
        segments = fname.split("_")
        assert len(segments) == 4
        strategy_name = segments[-2]

        df = pd.read_csv(fname)
        df.set_index("name", inplace=True)
        df.columns = [strategy_name]

        bt_results.append(df)

    all_results = pd.concat(bt_results, axis=1)
    print(all_results)
    all_results.to_csv("debug.csv", index_label="metric")


def combine_features():
    def _df_info(x, symbol, role):
        start_day, end_day = [x.index[idx].to_pydatetime().strftime("%Y%m%d") for idx in [0, -1]]
        print(f"[{symbol} {role}] '{start_day}'~'{end_day}' shape={x.shape}")

    symbol = "TSLA"

    dfs = []
    for role in ["train", "val"]:
        filename = f"features/{symbol}_{role}.csv"
        df = pd.read_csv(filename, index_col="Date", parse_dates=True)
        _df_info(df, symbol, role)
        dfs.append(df)

    whole_df = pd.concat(dfs, axis=0)
    _df_info(whole_df, symbol, "trainval")
    whole_df.to_csv(f"features/{symbol}_trainval.csv", index_label="Date")


def test_new_task():
    task = Task(stage="test", description="test", program=__file__, prev_task="01094950")
    print(f"symbol={task.symbol}")
    print(task.done())


def test_old_task():
    task = Task(task_id="02142333")
    print(task)

    parents = task.parents(include_self=True)
    for idx, ptask in enumerate(parents, start=1):
        print(f"[{idx}]-th parent ~ {ptask.stage} ~ {ptask.program}")

    ptask = task.parent(lambda t: t.stage == STAGE.MAKE_DATASET)
    print(ptask)


class TestWinGenerator:
    def __init__(
        self, n_steps, batch_size, window_size, feat_dims, feat_groups, weight_col, only_feature
    ) -> None:
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.window_size = window_size
        self.feat_dims = feat_dims
        self.feat_groups = feat_groups

        self._weight_col = weight_col
        self._only_feature = only_feature

        self.df = self.make_series_df()
        self.wg = WindowGenerator(
            win_size=window_size,
            all_cols=list(self.df.columns),
            feat_groups=feat_groups,
            only_feature=self._only_feature,
        )

    def make_series_df(self):
        data = []
        total_dims = sum(n for c, n in self.feat_dims)
        for idx in range(1, self.n_steps + 1):
            data.append([idx for _ in range(total_dims)])

        columns = [f"{c}{idx}" for c, n in self.feat_dims for idx in range(1, 1 + n)]

        data = pd.DataFrame(data, columns=columns)
        data["forward_move"] = -data.sum(axis=1)

        if self._weight_col is not None:
            data[self._weight_col] = -data["forward_move"] / 100

        print(data)
        return data

    def check(self, batch):
        if self._only_feature:
            features = batch
        else:
            features, label, weight = batch

        if not isinstance(features, dict):
            features = {"UNK": features}

        grp_checksum = []
        for grpname, grpdata in features.items():
            print(f"\n--- FEATURE GROUP[{grpname}]=\n{grpdata}")

            mode = self.feat_groups[grpname]["mode"]
            if mode == "window":
                # input grpdata: [batch_size, window_size, feat_dim]
                # reduce_sum: [batch_size, window_size]
                # [:,-1]: [batch_size]
                tail = tf.math.reduce_sum(grpdata, axis=-1)[:, -1]
            elif mode == "last":
                # input grpdata: [batch_size, feat_dim]
                # reduce_sum: [batch_size]
                tail = tf.math.reduce_sum(grpdata, axis=-1)
            else:
                raise ValueError(f"unknown mode={mode}")
            grp_checksum.append(tail)
        total_checksum = tf.add_n(grp_checksum)

        if not self._only_feature:
            tf.debugging.assert_equal(-total_checksum, tf.reshape(label, [-1]))
            print(f"\nlabel=\n{label}")

            if self._weight_col is not None:
                tf.debugging.assert_near(total_checksum / 100, tf.reshape(weight, [-1]))
            print(f"\nweight=\n{weight}")

    def test_splitwindow(self):
        batch = []
        offset = 0
        for bidx in range(self.batch_size):
            start_pos = offset + bidx
            end_pos = start_pos + self.window_size
            # batch.append(df.iloc[start_pos:end_pos, :].to_numpy())# float64
            batch.append(np.asarray(self.df.iloc[start_pos:end_pos, :], dtype=np.float32))  # float32

        batch = tf.stack(batch)
        print(batch)  # [batch_size, window_size, total_features]

        batch = self.wg.split_window(batch)
        self.check(batch)

    def test_makedataset(self):
        # here shuffle=True indicating shuffle the window within or across batches
        # but inside one window, datas (i.e., rows) are still chronologically ordered
        dataset = self.wg.make_dataset(self.df, batch_size=self.batch_size, shuffle=True)

        for bidx, batch in enumerate(dataset, start=1):
            print(f"\n********************* BATCH[{bidx}]")
            self.check(batch)


def test_discretize_sample_weights():
    task = Task(task_id="08091912")
    raw_features = task.load_raw_features()

    n_bins = 5
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")

    max_weight = 5
    sample_weights = discretizer.fit_transform(raw_features.loc[:, ["sample_weight"]])
    raw_features["sample_weight_bin"] = 1 + (max_weight - 1) * sample_weights / (n_bins - 1)

    bin_counts = raw_features["sample_weight_bin"].value_counts().to_dict()
    for weight, count in bin_counts.items():
        print(f"weight={weight}, count={count:6d}, ratio={count*100 / raw_features.shape[0]:.3f}%")


def describe_task():
    task = Task(task_id="08091912")
    task.description = "Model Predict + ATR position"

    TASK_MANAGER.update(task)
    TASK_MANAGER.save()


def test_bool_series():
    bs = pd.Series([1, -2, 6, -5, 2, 3]) > 0

    shift_bs = bs.shift(1)
    print(shift_bs)  # first element is NaN

    # print(~shift_bs) # CANNOT use ~ on first NaN

    print(shift_bs & bs)  # but NaN can & or / with other bool


def test_parent_tasks():
    task = Task(task_id="16090258")
    for idx, pt in enumerate(task.parents(include_self=True), start=1):
        print(f"\n************[-{idx}]: \n{pt}")


def test_build_model():
    configs = {
        "seq_len": 20,
        "rnn": [32, 32],
        "dense": [32, 32],
        "l2": 0.01,
    }
    builder = ImproveRnnWithIndicator(configs)
    model = builder.build()
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


def test_binary_search():
    a = [1, 22, 33, 44, 55]
    print(f"original array = {a}")

    for v in [-1, 1, 3, 0, 55, 5, 6, 88, 38]:
        pos = bisect_left(a, v)
        cpyarr = list(a)

        result = None
        if pos == len(a):
            cpyarr.append(v)
            result = f"missing, should append at tail, {cpyarr}"

        elif a[pos] == v:
            result = f"found at pos={pos}"
        else:
            cpyarr.insert(pos, v)
            result = f"missing, should insert at pos={pos}, {cpyarr}"

        print(f"v=[{v}] ---> {result}")


def test_bisect_mapping():
    d = {
        "10p": 0.32217341661453247,
        "20p": 0.3365685045719147,
        "30p": 0.34630629420280457,
        "40p": 0.35559776425361633,
        "50p": 0.36692947149276733,
        "60p": 0.37756240367889404,
        "70p": 0.39038777351379395,
        "80p": 0.40586602687835693,
        "90p": 0.4316391050815582,
    }

    keys = [0]
    for _, v in d.items():
        keys.append(v)
    keys.append(1)
    assert len(keys) % 2 == 1, "length should be odd"
    middle = len(keys) // 2

    values = []
    for idx in range(len(keys)):
        v = idx - middle
        if idx <= middle:
            v -= 1
        values.append(abs(v / middle))

    df = pd.DataFrame({"key": keys, "value": values})
    print(df)

    def _search_(search_k):
        pos = bisect_left(keys, search_k)
        print(f"k={search_k:.4f} >>>>>> v={values[pos]}")

    for j, v in enumerate(d.values(), start=1):
        print(f"\n----------- [{j}] testing bound={v:.4f}")
        _search_(v - 0.001)
        _search_(v + 0.001)


def temp():
    d = {
        "Close": "LOG_SCALING",
        "Volume": "LOG_SCALING",
        "atr_10d": "LOG_SCALING",
        "atr_20d": "LOG_SCALING",
        "atr_5d": "LOG_SCALING",
        "c-o": "SCALING",
        "chaikin": "SCALING",
        "forward_change": "USER_DEFINE",
        "forward_move": "COPY",
        "h-l": "LOG_SCALING",
        "is_boll10_dead": "COPY",
        "is_boll10_gold": "COPY",
        "is_boll20_dead": "COPY",
        "is_boll20_gold": "COPY",
        "is_macd_dead": "COPY",
        "is_macd_gold": "COPY",
        "is_pricema_dead": "COPY",
        "is_pricema_gold": "COPY",
        "is_rsi5_dead": "COPY",
        "is_rsi5_gold": "COPY",
        "macdhist": "SCALING",
        "macdsignal": "SCALING",
        "obv": "SCALING",
        "price_10d_sma": "LOG_SCALING",
        "price_10d_std": "LOG_SCALING",
        "price_20d_sma": "LOG_SCALING",
        "price_20d_std": "LOG_SCALING",
        "price_5d_sma": "LOG_SCALING",
        "price_5d_std": "LOG_SCALING",
        "return_b10d": "SCALING",
        "return_b10d_std": "SCALING",
        "return_b1d": "SCALING",
        "return_b20d": "SCALING",
        "return_b20d_std": "SCALING",
        "return_b5d": "SCALING",
        "return_b5d_std": "SCALING",
        "rsi_10d": "SCALING",
        "rsi_20d": "SCALING",
        "rsi_5d": "SCALING",
        "vlm_10d_avg": "LOG_SCALING",
        "vlm_10d_std": "LOG_SCALING",
        "vlm_20d_avg": "LOG_SCALING",
        "vlm_20d_std": "LOG_SCALING",
        "vlm_5d_avg": "LOG_SCALING",
        "vlm_5d_std": "LOG_SCALING",
    }

    nd = defaultdict(list)
    for k, v in d.items():
        nd[v].append(k)

    for k, v in nd.items():
        print(f"\n---------- {k}: {len(v)} features")
        print(", ".join(v))


def chaikin_ad_line(df, period):
    # Calculate the money flow volume.
    money_flow_volume = (df["close"] - df["low"]) - (df["high"] - df["close"]) / (
        df["high"] - df["low"]
    ) * df["volume"]

    # Calculate the Chaikin A/D Line.
    chaikin_ad_line = pd.Series(np.cumsum(money_flow_volume), name="chaikin_ad_line", copy=False)
    chaikin_ad_line = chaikin_ad_line / df["volume"].sum()

    return chaikin_ad_line


def test_reformat_model_metrics():
    fname = "evaluations/dataset_15160538_models_eval.csv"
    df = pd.read_csv(fname, index_col="config or metric")

    useful_indices = [
        f"{metric}#{role}" for metric in ["AUC", "score_stddev"] for role in ["train", "val", "test"]
    ]
    all_metrics = df.loc[useful_indices, :].to_dict(orient="dict")

    new_metrics = []
    for modelname, one_old_metrics in all_metrics.items():
        for old_metric_name, metric_value in one_old_metrics.items():
            metric_name, role = old_metric_name.split("#")

            one_new_metric = {
                "model": modelname,
                "metric": metric_name,
                "role": role,
                "value": float(metric_value),  # data in original dataframe is typed as object
            }
            new_metrics.append(one_new_metric)

    new_metrics = pd.DataFrame(new_metrics)
    print(new_metrics)

    # --------------- plot
    auc_metrics = new_metrics.loc[new_metrics["metric"] == "AUC", :]
    sns.barplot(data=auc_metrics, x="model", y="value", hue="role")
    plt.show()


if __name__ == "__main__":
    # test_build_lstm()
    # test_load_features()
    # test_timeseries_generator()
    # test_get_lstm_weights()
    # combine_backtest_results(symbol="TSLA", role="test")
    # combine_features()
    # test_new_task()
    # test_old_task()
    # test_discretize_sample_weights()
    # test_bool_series()
    # test_parent_tasks()
    # test_build_model()
    # test_binary_search()
    # test_bisect_mapping()
    # test_reformat_model_metrics()

    # test_wingen = TestWinGenerator(
    #     n_steps=11,
    #     batch_size=4,
    #     window_size=3,
    #     feat_dims=[("A", 3), ("B", 2), ("C", 1)],
    #     feat_groups={
    #         "G1": {"mode": "window", "feat_names": ["A1",  "C1"]},
    #         "G2": {"mode": "last", "feat_names": ["A2", "A3", "B1", "B2"]},
    #     },
    #     weight_col="sample_weight",
    #     only_feature=False,
    # )
    # # test_wingen.test_splitwindow()
    # test_wingen.test_makedataset()

    # describe_task()
    temp()
    pass
