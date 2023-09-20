import argparse
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from task_manager import Task, STAGE
import strategy
from backtest_evt import JobEventBacktest
from pathlib import Path
import time
from copy import copy

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

DEFAULT_INIT_CASH = 1000000
DEFAULT_COMMISSION = 0.002


class JobBatchRunBacktests:
    def __init__(self, pred_taskid) -> None:
        self._pred_taskid = pred_taskid
        self._bt_tasks = defaultdict(dict)

    def backtest(self, strategy_name, sig_class, pos_class, role, config):
        task = Task(
            prev_task=self._pred_taskid,
            stage=STAGE.EVT_BACKTEST,
            config=config,
        )

        task_id = JobEventBacktest(sig_class=sig_class, pos_class=pos_class, role=role, task=task).main()
        self._bt_tasks[strategy_name][role] = task_id

    def disp_pred_distribution(self):
        task = Task(task_id=self._pred_taskid)
        assert task.stage == STAGE.PREDICT
        print(json.dumps(task.outputs, indent=4))

        fig, ax = plt.subplots(figsize=(20, 10))
        for idx, (role, color) in enumerate(
            [("train", "steelblue"), ("val", "orange"), ("test", "green")], start=1
        ):
            fname = task.bt_input(role)
            df = pd.read_csv(fname, parse_dates=True, index_col="Date")

            pred_probas = df["pred_up_proba"]
            sns.distplot(pred_probas, label=role, color=color)

            median_proba = pred_probas.median()
            plt.axvline(x=median_proba, ymin=0, ymax=12, linestyle="--", color=color)
            plt.text(x=median_proba - 0.01, y=9 - idx, color=color, s=f"{role} median={median_proba:.2f}")

        plt.legend()
        plt.grid()
        plt.title("model prediction distribution")
        plt.show()

    def __save(self, role, new_experiments, overwrite):
        outfname = f"eval_strategy/pred_{self._pred_taskid}_strategies_{role}.csv"
        index_label = "config or metric"

        experiments = {}
        if (not overwrite) and Path(outfname).exists():
            experiments = pd.read_csv(outfname, index_col=index_label)
            experiments = experiments.to_dict(orient="series")

        experiments.update(new_experiments)
        experiments = pd.DataFrame(experiments)
        experiments.to_csv(outfname, index_label=index_label)
        print(f"\n{'OVERWRITE' if overwrite else 'APPEND'} '{role}' experiments in '{outfname}'")

    def summarize_experiments(self, overwrite):
        # ----------------- collect
        new_experiments = defaultdict(dict)
        for strategy_name, task_ids in self._bt_tasks.items():
            for role, task_id in task_ids.items():
                task = Task(task_id=task_id)
                assert task.prev_task == self._pred_taskid

                with open(task.bt_output(role, "info.json"), "rt") as fin:
                    d = json.load(fin)
                    performance = d[0]["outputs"]["performance"]

                info = {"00_bt_task": task_id}
                info.update({("01_" + k): v for k, v in task.config.items() if k != "proba_percentiles"})
                info.update({f"{BT_METRIC_ORDER[k]+10}_{k}": v for k, v in performance.items()})

                new_experiments[role][strategy_name] = pd.Series(info)

        # ----------------- save
        for role, _new_exmts in new_experiments.items():
            self.__save(role=role, new_experiments=_new_exmts, overwrite=overwrite)

    def __backtest_buyhold(self):
        config = {
            "init_cash": DEFAULT_INIT_CASH,
            "commission": 0.0,  # simulate buy & hold
        }
        for role in ["val", "test"]:
            self.backtest(
                strategy_name="buy & hold",
                sig_class=strategy.DummySignal,  # just a place holder
                pos_class=strategy.SellBuyEveryday,
                role=role,
                config=config,
            )

    def __backtest_sma(self):
        config = {
            "init_cash": DEFAULT_INIT_CASH,
            "commission": DEFAULT_COMMISSION,
        }
        for role in ["val", "test"]:
            self.backtest(
                strategy_name="SMA+SimplePosition",
                sig_class=strategy.SmaSignal,
                pos_class=strategy.SimplePosition,
                role=role,
                config=config,
            )

    def __backtest_model(self, thresholds):
        config_template = {
            "init_cash": DEFAULT_INIT_CASH,
            "commission": DEFAULT_COMMISSION,
            "risk_ratio": 0.03,
            "n_atr": 3,
            "allow_scale_in": True,
            "stop_loss": False,
            "take_profit": False,
            "adjust_size_by_proba": False,
        }

        update_atr_cfgs = {
            "Model+AtrPos": {},
            "Model+AtrPos+AdjustSize": {
                "adjust_size_by_proba": True,
                "signal_strength_scale": 0.9,
            },
            "Model+AtrPos+SLTP": {
                "stop_loss": True,
                "take_profit": True,
            },
        }

        for role, (down_threshold, up_threshold) in thresholds.items():
            new_config = copy(config_template)
            new_config["down_threshold"] = down_threshold
            new_config["up_threshold"] = up_threshold

            self.backtest(
                strategy_name="Model+SimplePos",
                sig_class=strategy.ModelSignal,
                pos_class=strategy.SimplePosition,
                role=role,
                config=new_config,
            )

            for strategy_name, changes in update_atr_cfgs.items():
                new_config.update(changes)

                self.backtest(
                    strategy_name=strategy_name,
                    sig_class=strategy.ModelSignal,
                    pos_class=strategy.AtrPosition,
                    role=role,
                    config=new_config,
                )

    def run_backtests(self, thresholds, overwrite):
        self.__backtest_buyhold()
        self.__backtest_sma()
        self.__backtest_model(thresholds)
        self.summarize_experiments(overwrite)

    def __collect_metrics(self, role):
        fname = f"eval_strategy/pred_{self._pred_taskid}_strategies_{role}.csv"
        df = pd.read_csv(fname, index_col="config or metric", parse_dates=True)

        # --------------- reformat the data
        useful_indices = [r for r in df.index if not r.startswith("0")]
        all_metrics = df.loc[useful_indices, :].to_dict(orient="dict")

        new_metrics = []
        for strategy_name, one_old_metrics in all_metrics.items():
            for metric_name, metric_value in one_old_metrics.items():
                if "days" in metric_value:
                    metric_value = metric_value.split("days")[0]
                try:
                    metric_value = float(metric_value)
                except ValueError:
                    continue  # datetime cannot cast, ignore
                else:
                    one_new_metric = {
                        "strategy": strategy_name,
                        "metric": metric_name[3:],
                        "role": role,
                        "value": metric_value,
                    }
                    new_metrics.append(one_new_metric)

        return new_metrics

    def compare_strategies(self):
        metrics = []
        for role in ["val", "test"]:
            metrics.extend(self.__collect_metrics(role))
        metrics = pd.DataFrame(metrics)

        pics = [
            [
                "Sharpe Ratio",
                "Max. Drawdown [%]",
                "Win Rate [%]",
                "Profit Factor",
            ],
            [
                "SQN",
                "Return (Ann.) [%]",
                "Max. Drawdown Duration",
                "Volatility (Ann.) [%]",
            ],
            [
                "Sortino Ratio",
                "Calmar Ratio",
                "Best Trade [%]",
                "Worst Trade [%]",
            ],
        ]
        for useful_metrics in pics:
            nrows = 2
            ncols = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[15, 15])
            for idx, metric_name in enumerate(useful_metrics):
                ax = axes[idx // ncols, idx % ncols]
                sub_metrics = metrics.loc[metrics["metric"] == metric_name, :]
                sns.barplot(data=sub_metrics, x="strategy", y="value", hue="role", ax=ax)

                ax.margins(y=0.1)  # make room for the labels
                for bars in ax.containers:
                    ax.bar_label(bars, fmt="%.2f")

                ax.set_title(metric_name)
                ax.grid()
                ax.xaxis.set_tick_params(rotation=45, labelsize=8)
                ax.tick_params(axis="x", direction="in", pad=-10)
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(), ha="right")

            fig.tight_layout()
            plt.show()


def adjust_thresholds():
    pred_taskid = "16165705"
    sig_class = strategy.ModelSignal
    pos_class = strategy.AtrPosition

    role = "test"
    down_threshold = 0.32
    up_threshold = 0.35

    config = {
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
        "init_cash": 1000000,
        "commission": 0.002,
        "risk_ratio": 0.03,
        "n_atr": 3,
        "allow_scale_in": True,
        "stop_loss": False,
        "take_profit": False,
        "adjust_size_by_proba": False,
        "signal_strength_scale": 0.9,
    }

    strategy_name = f"Model+Atr_{int(down_threshold*100)}_{int(up_threshold*100)}"
    job = JobBatchRunBacktests(pred_taskid=pred_taskid)
    job.backtest(
        strategy_name=strategy_name, sig_class=sig_class, pos_class=pos_class, role=role, config=config
    )
    job.summarize_experiments(overwrite=False)


def auto_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task", required=False)
    parser.add_argument("-j", "--job", required=True)
    parser.add_argument("-ow", "--overwrite", type=int)
    args = parser.parse_args()

    job = JobBatchRunBacktests(pred_taskid=args.prev_task)
    if args.job == "distribution":
        job.disp_pred_distribution()
    elif args.job == "backtest":
        if args.overwrite is None:
            raise ValueError("must provide overwrite arguments")
        job.run_backtests(overwrite=(args.overwrite == 1))
    else:
        raise ValueError(f"unknown job={args.job}")


def manual_main():
    pred_taskid = "16165705"
    job = JobBatchRunBacktests(pred_taskid=pred_taskid)
    # job.disp_pred_distribution()

    # thresholds = {"val": [0.33, 0.37], "test": [0.32, 0.35]}
    # job.run_backtests(thresholds=thresholds, overwrite=True)

    job.compare_strategies()


if __name__ == "__main__":
    # adjust_thresholds()
    manual_main()
