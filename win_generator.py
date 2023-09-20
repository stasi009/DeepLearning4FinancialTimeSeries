from collections import defaultdict
import numpy as np
import tensorflow as tf


def all_or_onlyitem(d):
    if len(d) > 1:
        return d

    for v in d.values():
        return v  # return only item


class WindowGenerator:
    def __init__(self, win_size, all_cols, feat_groups, only_feature) -> None:
        self._win_size = win_size
        self._only_feature = only_feature

        self._all_cols = all_cols
        col2index = {c: idx for idx, c in enumerate(self._all_cols)}

        self._weight_index = col2index.get("sample_weight")
        self._label_index = col2index.get("forward_move")
        if self._label_index is None and not self._only_feature:
            raise ValueError("label column missing in train mode")

        # one input feature can appear in multiple groups
        self._feat_groups = {}
        for grpname, oldinfo in feat_groups.items():
            feat_indices = []
            for name in oldinfo["feat_names"]:
                if name == "forward_move" or name == "sample_weight":
                    raise ValueError("label leaked into training data")
                feat_indices.append(col2index[name])

            self._feat_groups[grpname] = {
                "mode": oldinfo["mode"],
                "feat_indices": feat_indices,
            }

    def split_window(self, batch):
        """
        data: [batch_size, window_size, total_columns]
        """
        # *************************** FEATURE
        features = {}
        for grpname, grpinfo in self._feat_groups.items():
            feat_indices = grpinfo["feat_indices"]
            # each element is [batch_size, window_size]
            feat_grp_data = [batch[:, :, findex] for findex in feat_indices]
            # features: [batch_size, window_size, len(feat_indices)]
            feat_grp_data = tf.stack(feat_grp_data, axis=-1)
            # make `tf.data.Datasets` are easier to inspect.
            feat_grp_data.set_shape([None, self._win_size, len(feat_indices)])

            if grpinfo["mode"] != "window":
                assert grpinfo["mode"] == "last"
                # last step in the window, [batch_size, len(feat_indices)]
                feat_grp_data = feat_grp_data[:, -1, :]
                feat_grp_data.set_shape([None, len(feat_indices)])

            features[grpname] = feat_grp_data

        if self._only_feature:
            return all_or_onlyitem(features)  # no need to provide label and weight for prediction

        # *************************** LABEL
        # last example in the batch provides the label
        label = tf.reshape(batch[:, -1, self._label_index], [-1, 1])
        label.set_shape([None, 1])

        # *************************** SAMPLE WEIGHT
        if self._weight_index is None:
            weight = tf.ones_like(label)
        else:
            # last example in the batch provides the weight
            weight = tf.reshape(batch[:, -1, self._weight_index], [-1, 1])
        weight.set_shape([None, 1])

        # Model.fit expectes input dataset Should return
        # a tuple of either (inputs, targets) or (inputs, targets, sample_weights)
        # However, features & label can be a dictionary for multiple inputs or multiple outputs
        return all_or_onlyitem(features), label, weight

    def make_dataset(self, df, batch_size, shuffle):
        """
        - df: index is datetime
        - if shuffle=True indicating shuffle the window within or across batches,
        but inside one window, datas (i.e., rows) are still chronologically ordered
        """
        assert np.all(df.columns == self._all_cols), "columns mismatch"
        data = np.asarray(df, dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self._win_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        return ds.map(self.split_window)
