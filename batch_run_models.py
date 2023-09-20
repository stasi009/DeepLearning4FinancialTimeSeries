import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train import JobTrain
from predict_eval import JobPredict
import tensorflow as tf
import random
from model import BaseModel, SimpleRNN, ImproveRnnWithIndicators
from task_manager import Task, STAGE
from copy import copy
from pathlib import Path
import time

REALNUM_FEATS = [
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

INDICATOR_FEATS = [
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
]


class JobBatchRunModels:
    def __init__(self, dataset_taskid) -> None:
        self._dataset_taskid = dataset_taskid
        self._pred_tasks = {}
        self._seed = None

    def _run(self, exmt_name, feat_group, configs, model_class):
        configs["feature_group"] = feat_group
        configs["rand_seed"] = self._seed
        configs["model"] = model_class.__name__

        train_task = Task(prev_task=self._dataset_taskid, stage=STAGE.TRAIN, config=configs)
        model = model_class(configs)
        # no need to set seed again in separate tasks again
        train_job = JobTrain(model=model.build(), task=train_task)
        train_taskid = train_job.main(set_seed=False, refit=False)

        pred_task = Task(prev_task=train_taskid, stage=STAGE.PREDICT)
        self._pred_tasks[exmt_name] = JobPredict(pred_task).main()

    def run_simple_rnn(self):
        feat_group = {"feature": {"mode": "window", "feat_names": REALNUM_FEATS}}

        baseline_configs = {
            "seq_len": 20,
            "rnn": [64],
            "dense": [64],
            "epochs": 100,
            "earlystop_patience": 10,
            "l2": 0.01,
        }

        update_configs = {
            "baseline": {},
            "no_regularization": {"l2": 0},
            "longer_window": {"seq_len": 40},
            "smaller_layer_32": {
                "rnn": [32],
                "dense": [32],
            },
            "larger_layer_128": {
                "rnn": [128],
                "dense": [128],
            },
            "double_layers_32": {
                "rnn": [32, 32],
                "dense": [32, 32],
            },
        }

        for exmt_name, changes in update_configs.items():
            print(f"\n**************** RUN EXPERIMENT [{exmt_name}]")

            new_configs = copy(baseline_configs)
            new_configs.update(changes)

            self._run(
                exmt_name=exmt_name, feat_group=feat_group, configs=new_configs, model_class=SimpleRNN
            )

    def run_improved_rnn(self):
        feat_group = {
            "realnum": {"mode": "window", "feat_names": REALNUM_FEATS},
            "indicator": {"mode": "last", "feat_names": INDICATOR_FEATS},
        }

        baseline_configs = {
            "seq_len": 20,
            "rnn": [64],
            "dense": [64],
            "epochs": 100,
            "earlystop_patience": 10,
            "l2": 0.01,
        }

        update_configs = {
            "improved_rnn": {},
            "impv_rnn_longer_win": {"seq_len": 40},
        }

        for exmt_name, changes in update_configs.items():
            print(f"\n**************** RUN EXPERIMENT [{exmt_name}]")

            new_configs = copy(baseline_configs)
            new_configs.update(changes)

            self._run(
                exmt_name=exmt_name,
                feat_group=feat_group,
                configs=new_configs,
                model_class=ImproveRnnWithIndicators,
            )

    def __save(self, new_summary, overwrite):
        outfname = f"eval_model/dataset_{self._dataset_taskid}_models_eval.csv"
        index_label = "config or metric"

        summary = {}
        if (not overwrite) and Path(outfname).exists():
            summary = pd.read_csv(outfname, index_col=index_label)
            summary = summary.to_dict(orient="series")

        summary.update(new_summary)
        summary = pd.DataFrame(summary)
        summary.to_csv(outfname, index_label=index_label)

        print(f"\n{'OVERWRITE' if overwrite else 'APPEND'} summary in '{outfname}'")

    def summarize(self, overwrite):
        summary = defaultdict(dict)

        for model_name, taskid in self._pred_tasks.items():
            pred_task = Task(task_id=taskid)
            assert (
                pred_task.stage == STAGE.PREDICT
            ), f"model={model_name},pred_task={taskid},stage={pred_task.stage}"

            train_task = Task(task_id=pred_task.prev_task)
            assert (
                train_task.prev_task == self._dataset_taskid
            ), f"model={model_name},train_task={train_task.task_id},not follow dataset[{self._dataset_taskid}]"

            # ---------- fill input configs
            info = {"pred_task": taskid}
            for k, v in train_task.config.items():
                if k == "feature_group":
                    v = [
                        f"{grpname} has {len(grpinfo['feat_names'])} {grpinfo['mode']} feats"
                        for grpname, grpinfo in v.items()
                    ]
                    v = ", ".join(v)
                if k == "dense" or k == "rnn":
                    v = ",".join(str(n) for n in v)
                info[k] = v
            info["*********"] = "*********"  # separator

            # ---------- fill output metrics
            for metric in ["AUC", "score_stddev"]:
                for role in ["train", "val", "test"]:
                    info[f"{metric}#{role}"] = pred_task.outputs[metric][role]

            summary[model_name] = pd.Series(info)

        self.__save(summary, overwrite)

    def run_models(self, seed, overwrite):
        self._seed = seed
        if self._seed is None:
            self._seed = random.randint(1, 90000)
        tf.random.set_seed(self._seed)
        tf.keras.utils.set_random_seed(self._seed)

        self.run_simple_rnn()
        self.run_improved_rnn()

        self.summarize(overwrite)

    def compare_models(self):
        fname = f"eval_model/dataset_{self._dataset_taskid}_models_eval.csv"
        df = pd.read_csv(fname, index_col="config or metric")

        # --------------- reformat the data
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
        # print(new_metrics)

        # --------------- plot
        fig, axes = plt.subplots(nrows=2, figsize=[15, 15])
        for r, metric in enumerate(["AUC", "score_stddev"]):
            ax = axes[r]
            _metrics = new_metrics.loc[new_metrics["metric"] == metric, :]
            sns.barplot(data=_metrics, x="model", y="value", hue="role", ax=ax)

            ax.margins(y=0.1)  # make room for the labels
            for bars in ax.containers:
                ax.bar_label(bars, fmt="%.2f")

            ax.set_title(f"Model {metric}")
            ax.grid()

        plt.show()


def plot_model():
    # feat_group = {"feature": {"mode": "window", "feat_names": REALNUM_FEATS}}
    # model_class = SimpleRNN

    feat_group = {
        "realnum": {"mode": "window", "feat_names": REALNUM_FEATS},
        "indicator": {"mode": "last", "feat_names": INDICATOR_FEATS},
    }
    model_class = ImproveRnnWithIndicators

    configs = {
        "seq_len": 20,
        "rnn": [64],
        "dense": [64],
        "epochs": 100,
        "earlystop_patience": 10,
        "l2": 0.01,
        "feature_group": feat_group,
    }

    model = model_class(configs).build()
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task", required=True)
    parser.add_argument("-j", "--job", required=True)
    parser.add_argument("-s", "--rand_seed", type=int)
    parser.add_argument("-ow", "--overwrite", type=int)
    args = parser.parse_args()

    job = JobBatchRunModels(dataset_taskid=args.prev_task)
    if args.job == "run":
        if args.overwrite is None:
            raise ValueError("must provide overwrite arguments")
        job.run_models(seed=args.rand_seed, overwrite=(args.overwrite == 1))
    elif args.job == "compare":
        job.compare_models()
    else:
        raise ValueError(f"unknown job={args.job}")
