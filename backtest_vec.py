import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from task_manager import STAGE, Task


def make_longshort_positions(pred_probas, down_threshold, up_threshold):
    long_short_positions = []
    for proba in pred_probas:
        position = 0
        if proba >= up_threshold:
            position = 1
        elif proba <= down_threshold:
            position = -1
        long_short_positions.append(position)
    return long_short_positions


class JobVectorBacktest:
    def __init__(self, task: Task) -> None:
        self._task = task
        self._pred_task = Task(self._task.prev_task)
        assert self._pred_task.stage == STAGE.PREDICT

    def __forward_returns(self, strategy, pred_position, result):
        result[f"{strategy}_predret_position"] = pred_position

        # RHS old pred_position: today predict tomorrow
        # LHS new pred_position: after shift(1), yesterday predict today
        pred_position = pred_position.shift(1)

        # pred_position: yesterday make decision about today
        # return: today against yesterday
        result[f"{strategy}_predret_gross"] = pred_position * result["return_b1d"]
        result[f"{strategy}_predval_gross"] = result[f"{strategy}_predret_gross"].cumsum().apply(np.exp)

        # if today's position is different from yesterday, trade occur today
        # if position changes from -1 to 1, or 1 to -1, double the commission
        trades = pred_position.diff().fillna(0).abs()
        n_trades = (trades != 0).sum()
        print(
            f"{n_trades} trades in {len(pred_position)} days, ratio={n_trades / len(pred_position)*100:.2f}%"
        )

        result[f"{strategy}_predret_net"] = (
            result[f"{strategy}_predret_gross"] - trades * self._task.config["commission"]
        )
        result[f"{strategy}_predval_net"] = result[f"{strategy}_predret_net"].cumsum().apply(np.exp)

    def main(self):
        role = "test"
        infname = self._pred_task.vector_bt_file(role, "in")
        result = pd.read_csv(infname, index_col="Date", parse_dates=True)

        # ******************* LONG-ONLY STRATEGY
        pred_probas = result["pred_up_proba"]
        up_threshold, down_threshold = (self._task.config[k] for k in ["up_threshold", "down_threshold"])

        long_only_positions = (pred_probas > up_threshold).astype(int)
        self.__forward_returns(strategy="longonly", pred_position=long_only_positions, result=result)

        # ******************* LONG-SHORT STRATEGY
        long_short_positions = make_longshort_positions(
            pred_probas, down_threshold=down_threshold, up_threshold=up_threshold
        )
        self.__forward_returns(
            strategy="longshort",
            pred_position=pd.Series(long_short_positions, index=pred_probas.index),
            result=result,
        )

        # ******************* TASK DONE
        result.dropna(inplace=True)
        # calculate longhold net-value after dropna, because predict net-value is always NaN at first row
        result["longhold_val"] = result[f"return_b1d"].cumsum().apply(np.exp)

        outfname = self._task.vector_bt_file(role, "out")
        result.to_csv(outfname, index_label="Date")

        plot_cols = ["longhold_val", "longshort_predval_net"]
        result.loc[:, plot_cols].plot(grid=True)
        plt.show()

        return self._task.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task")
    parser.add_argument("-u", "--up_threshold", type=float)
    parser.add_argument("-d", "--down_threshold", type=float)
    args = parser.parse_args()

    task = Task(
        prev_task=args.prev_task if args.prev_task is not None else Task.latest_task().task_id,
        stage=STAGE.VEC_BACKTEST,
        config={
            "up_threshold": args.up_threshold,
            "down_threshold": args.down_threshold,
            "commission": 0.002,
        },
    )
    JobVectorBacktest(task).main()
