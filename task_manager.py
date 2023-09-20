import json
import pandas as pd
from datetime import datetime
from enum import Enum
import time


STAGE = Enum(
    "STAGE",
    ["ADD_FEATURE", "MAKE_DATASET", "TRAIN", "PREDICT", "VEC_BACKTEST", "EVT_BACKTEST", "SELECT_FEATURE"],
)


class TaskManager:
    def __init__(self) -> None:
        self.filename = "tasks_log.json"
        try:
            with open(self.filename, "rt") as fin:
                self._task_logs = json.load(fin)
        except (json.decoder.JSONDecodeError, FileNotFoundError):
            # first time open when the file is empty
            self._task_logs = {}

    def get(self, task_id, return_dict=False):
        info = self._task_logs.get(task_id)
        if info is None:
            return None

        if return_dict:
            return info

        task = Task(stage="")
        task.task_id = task_id  # set outside constructor, avoid looking up
        task.from_dict(info)

        return task

    def update(self, task, allow_overwrite=False):
        if (not allow_overwrite) and task.task_id in self._task_logs:
            raise KeyError(f"task[{task.task_id}] already exits")
        self._task_logs[task.task_id] = task.to_dict(include_id=False)

    def save(self):
        with open(self.filename, "wt") as fout:
            json.dump(self._task_logs, fout, sort_keys=True, indent=4)

    @property
    def latest(self):
        max_timestamp = max(list(self._task_logs.keys()))
        return self.get(max_timestamp)


TASK_MANAGER = TaskManager()


class Task:
    def __init__(
        self,
        task_id=None,
        stage=None,
        description="",
        prev_task=None,
        config={},
    ) -> None:
        self.outputs = {}

        if task_id is None:  # new task
            self.task_id = datetime.now().strftime("%d%H%M%S")
            self.description = description

            assert stage is not None, "must provide stage"
            self.stage = stage

            self.prev_task = prev_task
            self.config = config
        else:
            d = TASK_MANAGER.get(task_id=task_id, return_dict=True)
            assert d is not None
            self.task_id = task_id
            self.from_dict(d)

    def from_dict(self, info):
        self.stage = STAGE[info["stage"]]
        self.description = info.get("description", "")
        self.prev_task = info.get("prev_task")  # one task can have no prev_task
        self.config = info.get("config")
        self.outputs = info.get("outputs", {})

    def to_dict(self, include_id):
        d = {
            "stage": self.stage.name,
            "config": self.config,
        }
        if self.prev_task is not None:
            d["prev_task"] = self.prev_task

        if len(self.outputs) > 0:
            d["outputs"] = self.outputs

        if len(self.description) > 0:
            d["description"] = self.description

        if include_id:
            d["task_id"] = self.task_id

        return d

    def __str__(self) -> str:
        d = self.to_dict(include_id=True)
        return json.dumps(d, sort_keys=True, indent=4)

    def parents(self, include_self=False):
        _parents = []
        if include_self:
            _parents.append(self)

        task = self
        while task.prev_task is not None:
            task = Task(task_id=task.prev_task)
            _parents.append(task)
        return _parents

    def parent(self, match_func):
        task = self
        while task.prev_task is not None:
            task = Task(task_id=task.prev_task)
            if match_func(task):
                return task
        return None

    def done(self):
        TASK_MANAGER.update(self)
        TASK_MANAGER.save()

        print(f"\n************* TASK[{self.task_id}] DONE *************")
        print(str(self) + "\n")

        time.sleep(1)  # prevent next task has same taskid
        return self.task_id

    @staticmethod
    def latest_task():
        return TASK_MANAGER.latest

    # *********************** FILE OPERATION *********************** #
    @staticmethod
    def load_ohlcv(symbol):
        filename = f"datas/{symbol}.csv"
        df = pd.read_csv(filename, index_col="Date", parse_dates=True)

        df.drop(columns=["Close"], inplace=True)
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

        return df

    def save_raw_features(self, df):
        df.to_csv(f"features/raw_feat_{self.task_id}.csv", index_label="Date")

    def load_raw_features(self):
        filename = f"features/raw_feat_{self.task_id}.csv"
        return pd.read_csv(filename, index_col="Date", parse_dates=True)

    def save_dataset(self, role, df):
        df.to_csv(f"features/dataset_{self.task_id}_{role}.csv", index_label="Date")

    def load_dataset(self, role):
        filename = f"features/dataset_{self.task_id}_{role}.csv"
        df = pd.read_csv(filename, index_col="Date", parse_dates=True)

        start_day, end_day = (df.index[idx].to_pydatetime().strftime("%Y%m%d") for idx in [0, -1])
        print(
            f"[{role}] '{start_day}'~'{end_day}' shape={df.shape} positive-rate={df['forward_move'].mean():.3f}"
        )

        return df

    @property
    def keras_model(self):
        return f"models/{self.task_id}.keras"

    @property
    def refitted_keras_model(self):
        return f"models/{self.task_id}_refitted.keras"

    @property
    def keras_log(self):
        return f"logs/{self.task_id}"

    def vector_bt_file(self, role, direction):
        return f"eval_model/{self.task_id}_vecbt_{role}_{direction}.csv"

    def bt_input(self, role):
        return f"eval_model/{self.task_id}_evtbt_{role}_in.csv"

    def bt_output(self, role, suffix):
        return f"eval_strategy/{self.task_id}_evtbt_{role}_{suffix}"
