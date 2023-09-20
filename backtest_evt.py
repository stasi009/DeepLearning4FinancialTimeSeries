import argparse
import json
from backtesting import Backtest
import pandas as pd
from task_manager import Task, STAGE
import strategy


class JobEventBacktest:
    def __init__(self, sig_class, pos_class, role, task: Task) -> None:
        self._task = task
        self._role = role
        self._sig_class = sig_class
        self._pos_class = pos_class

        self._predict_task = Task(self._task.prev_task)
        assert self._predict_task.stage == STAGE.PREDICT

    def archive(self, bt_results):
        self._task.config["role"] = self._role
        self._task.config["sig_class"] = self._sig_class.__name__
        self._task.config["pos_class"] = self._pos_class.__name__
        performance = {
            k: str(v) if isinstance(v, pd.Timedelta) or isinstance(v, pd.Timestamp) else v
            for k, v in bt_results.to_dict().items()
        }
        self._task.outputs["performance"] = performance

        task_chain = [task.to_dict(include_id=True) for task in self._task.parents(include_self=True)]
        with open(self._task.bt_output(self._role, "info.json"), "wt") as fout:
            json.dump(task_chain, fout, indent=4)

    def main(self):
        # ***************** run backtest
        bt_input = pd.read_csv(self._predict_task.bt_input(self._role), index_col="Date", parse_dates=True)

        backtest = Backtest(
            data=bt_input,
            strategy=strategy.MyStrategy,
            cash=self._task.config.get("init_cash", 10000),
            commission=self._task.config.get("commission", 0.002),
        )

        bt_results = backtest.run(
            config=self._task.config, sig_class=self._sig_class, pos_class=self._pos_class
        )

        # ***************** display result
        valid_metrics = [s for s in bt_results.index if not s.startswith("_")]
        bt_results = bt_results.loc[valid_metrics]
        bt_results.name = "value"
        print(bt_results)

        # superimpose=False, don't draw coarse-grained candlestick
        backtest.plot(filename=self._task.bt_output(self._role, "plot.html"), superimpose=False)

        # ***************** archive
        self.archive(bt_results)
        return self._task.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task")
    parser.add_argument("-r", "--role", required=True)
    parser.add_argument("-u", "--up_threshold", type=float)
    parser.add_argument("-d", "--down_threshold", type=float)
    args = parser.parse_args()

    sig_class = strategy.ModelSignal
    pos_class = strategy.AtrPosition
    config = {
        "up_threshold": args.up_threshold,
        "down_threshold": args.down_threshold,
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

    task = Task(
        prev_task=args.prev_task if args.prev_task is not None else Task.latest_task().task_id,
        stage=STAGE.EVT_BACKTEST,
        config=config,
    )
    JobEventBacktest(sig_class=sig_class, pos_class=pos_class, role=args.role, task=task).main()
