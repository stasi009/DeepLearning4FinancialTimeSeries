import json
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from task_manager import Task, STAGE
from copy import copy
import logging
# from ydata_profiling import ProfileReport

BT_METRICS = [
    "Start",
    "End",
    "Duration",
    "Exposure Time [%]",
    "Equity Final [$]",
    "Equity Peak [$]",
    "Return [%]",
    "Buy & Hold Return [%]",
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Max. Drawdown Duration",
    "Avg. Drawdown Duration",
    "# Trades",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg. Trade [%]",
    "Max. Trade Duration",
    "Avg. Trade Duration",
    "Profit Factor",
    "Expectancy [%]",
    "SQN",
]

BT_METRIC_ORDER = {name: idx for idx, name in enumerate(BT_METRICS, start=1)}


def profiling_raw_features():
    task_id = "14053722"
    task = Task(task_id=task_id)
    raw_feats = task.load_raw_features()
    report = ProfileReport(raw_feats)
    report.to_file(f"features/rawfeat_{task_id}_profile_report.html")


def plot_raw_data(symbol):
    df = Task.load_ohlcv(symbol)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=[15, 5])

    for ax, col in zip(axes, ["Close", "Volume"]):
        ax.plot(df.index, df[col])
        ax.set_title(f"{symbol}:{col}")
        ax.set_xlabel("date")
        ax.set_ylabel(col)

        ax.grid()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.subplots_adjust(wspace=0, hspace=0.1)
    fig.tight_layout()


def plot_rawfeat_timeseries(task_id, feats, nrows, ncols):
    add_feat_task = Task(task_id=task_id)
    df = add_feat_task.load_raw_features()
    train_lastday, val_lastday = [df.index[int(b * df.shape[0])] for b in [0.5, 0.75]]

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=[15, 15])

    for idx, feat in enumerate(feats):
        ax = axes[idx // ncols, idx % ncols]

        ax.plot(df.index, df[feat], label=feat, color="k")
        ax.set_title(feat)
        ax.set_xlabel("date")
        ax.set_ylabel(feat)
        ax.grid()

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        ax.xaxis.set_tick_params(rotation=45, labelsize=8)
        # ax.tick_params(axis="x", direction="in", pad=-10)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), ha="right")

        # fill rectangle
        ymin, ymax = ax.get_ylim()
        for color, condition, label in [
            ("steelblue", df.index <= train_lastday, "train"),
            ("orangered", (df.index > train_lastday) & (df.index <= val_lastday), "validation"),
            ("darkgreen", df.index > val_lastday, "test"),
        ]:
            ax.fill_between(df.index, ymin, ymax, where=condition, color=color, alpha=0.3, label=label)

        ax.legend()

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()


class FeaturePlot:
    def __init__(self, task_id) -> None:
        self._task = Task(task_id=task_id)
        raw_feats = self._task.load_raw_features()

        splits = [2, 1, 1]
        splits = [x / sum(splits) for x in splits]
        train_size = int(raw_feats.shape[0] * splits[0])
        train_rawfeats = raw_feats.iloc[:train_size, :]

        self._realnum_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "c-o",
            "h-l",
            "macd",
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
            "price_5d_std",
            "price_5d_ewma",
            "price_10d_sma",
            "price_10d_std",
            "price_10d_ewma",
            "price_20d_sma",
            "price_20d_std",
            "price_20d_ewma",
            "rsi_5d",
            "rsi_10d",
            "rsi_20d",
            "atr_5d",
            "atr_10d",
            "atr_20d",
        ]
        self._train_realnum_feats = train_rawfeats.loc[:, self._realnum_cols]

        self._ind_cols = [
            "is_macd_gold",
            "is_macd_dead",
            "is_pricema_gold",
            "is_pricema_dead",
            "is_rsi5_gold",
            "is_rsi5_dead",
            "is_rsi10_gold",
            "is_rsi10_dead",
            "is_rsi20_gold",
            "is_rsi20_dead",
            "is_boll5_gold",
            "is_boll5_dead",
            "is_boll10_gold",
            "is_boll10_dead",
            "is_boll20_gold",
            "is_boll20_dead",
        ]
        self._train_ind_feats = train_rawfeats.loc[:, self._ind_cols]

    def indicator_mean_bar(self):
        mean_each_ind = self._train_ind_feats.mean(axis=0)
        mean_each_ind.sort_values(ascending=False, inplace=True)

        fig = plt.figure()
        # mean_each_ind.plot.bar()
        sns.barplot(x=mean_each_ind, y=mean_each_ind.index, palette="muted")

        plt.grid()
        plt.title("mean of indicator features")
        # fig.tight_layout()
        plt.show()

    def realnum_histogram(self):
        nrows, ncols = 8, 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))

        for idx, colname in enumerate(self._realnum_cols):
            series = self._train_realnum_feats.loc[:, colname]
            ax = axes[idx // ncols, idx % ncols]
            sns.distplot(series, label=colname, ax=ax)
            ax.set_title(colname)
            ax.set_xlabel(None)
            ax.grid()

        plt.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.show()

    def realnum_corr_heatmap(self):
        correlation = self._train_realnum_feats.corr()

        # -------- find most correlated feature pairs
        corr_threshold = 0.9
        feat_names = list(correlation.columns)
        correlated_pairs = []
        for i in range(correlation.shape[0]):
            correlated = {"feature": feat_names[i]}
            counter = 0
            for j in range(correlation.shape[0]):
                if j != i:
                    cc = correlation.iloc[i, j]
                    if abs(cc) >= corr_threshold:
                        counter += 1
                        correlated[f"CR{counter}"] = f"{feat_names[j]}:{cc:.2f}"

            correlated_pairs.append(correlated)

        correlated_pairs = pd.DataFrame(correlated_pairs)
        correlated_pairs.set_index("feature", inplace=True)
        correlated_pairs.to_csv("feat_correlation.csv", index_label="feature")

        # -------- plot heatmap
        fig, ax = plt.subplots(figsize=(20, 20))
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            correlation,
            mask=mask,
            annot=True,
            annot_kws={"size": 9},
            fmt="0.2f",
            linewidths=0.4,
            square=False,
            cbar=True,
            cmap=cmap,
            ax=ax,
        )
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    # profiling_raw_features()
    plot_rawfeat_timeseries(
        "15160403", feats=["return_b1d", "Close", "Volume", "macdhist"], nrows=2, ncols=2
    )
    # FeaturePlot(task_id="15160403").indicator_mean_bar()
    # FeaturePlot(task_id="15160403").realnum_corr_heatmap()
    # compare_strategies()
    pass
