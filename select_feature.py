from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import argparse
from task_manager import Task, STAGE
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import shap


def separate_xy(df):
    y = df["forward_move"]
    X = df.drop(columns=["forward_move", "forward_change"])
    return X, y


class JobSelectFeatures:
    def __init__(self, task: Task) -> None:
        self._task = task
        self._prev_task = Task(task_id=self._task.prev_task)
        assert self._prev_task.stage == STAGE.ADD_FEATURE

        train_df, val_df = self._load_data()
        self._Xtrain, self._ytrain = separate_xy(train_df)
        self._Xval, self._yval = separate_xy(val_df)

    def _load_data(self):
        df = self._prev_task.load_raw_features()
        total_examples = df.shape[0]

        splits = self._task.config["splits"]
        splits = [x / sum(splits) for x in splits]

        train_size = int(total_examples * splits[0])
        val_size = int(total_examples * splits[1])

        train_df = df.iloc[:train_size, :]
        val_df = df.iloc[train_size : (train_size + val_size), :]

        return train_df, val_df

    def _save_feat_importances(self, name, importances):
        print(importances)
        self._task.outputs[f"{name}_feat_importances"] = importances.to_dict()

    def explain_by_decisiontree(self):
        dt = DecisionTreeClassifier()
        dt.fit(X=self._Xtrain, y=self._ytrain)

        feat_importances = pd.Series(dt.feature_importances_, index=dt.feature_names_in_)
        feat_importances.sort_values(ascending=False, inplace=True)
        self._save_feat_importances("dt", feat_importances)

        # tedious, cannot see clearly
        # print(export_text(dt, feature_names=list(dt.feature_names_in_)))
        # plot_tree(dt,max_depth=5,feature_names=feat_importances.index)
        # plt.show()

    def explain_by_gbdt(self):
        feat_names = list(self._Xtrain.columns)

        gbm = xgb.XGBClassifier(max_depth=10, n_estimators=50, early_stopping_rounds=5, eval_metric="auc")
        gbm.fit(
            self._Xtrain,
            self._ytrain,
            eval_set=[(self._Xval, self._yval)],
        )

        feat_importances = pd.Series(gbm.feature_importances_, index=feat_names)
        feat_importances.sort_values(ascending=False, inplace=True)
        self._save_feat_importances("gbdt", feat_importances)

        explainer = shap.Explainer(gbm)
        shap_values = explainer(self._Xtrain)
        shap.plots.bar(shap_values,max_display=40)        
        shap.plots.beeswarm(shap_values, max_display=40)

    def main(self):
        self.explain_by_decisiontree()
        self.explain_by_gbdt()
        return self._task.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--prev_task")
    args = parser.parse_args()

    task = Task(
        prev_task=args.prev_task if args.prev_task is not None else Task.latest_task().task_id,
        stage=STAGE.SELECT_FEATURE,
        config={
            "splits": [7, 1.5, 1.5],
        },
    )
    job = JobSelectFeatures(task)
    job.main()
