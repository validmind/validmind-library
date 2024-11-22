# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Union

from sklearn.ensemble import IsolationForest
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("time_series_data", "forecasting", "statistical_test", "visualization")
@tasks("regression")
def MyIsolationForest(
    dataset: VMDataset,
    random_state: int = 0,
    contamination: float = 0.1,
    features_columns: Union[List[str], None] = None,
):
    """
    The Isolation Forest test is an algorithm used for anomaly detection in datasets. It is based
    on the concept of isolating anomalies rather than identifying normal data points. The test builds an ensemble
    of isolation trees, which are binary trees created by randomly selecting features and splitting the data based
    on random thresholds.

    The main idea behind the Isolation Forest test is that anomalies are likely to be isolated quickly in these trees
    compared to normal instances. Anomalies are expected to have shorter average path lengths in the trees,
    as they are different from the majority of the data points.

    It's important to note that the Isolation Forest test assumes anomalies are less frequent and have different properties
    compared to normal instances. However, it may not be as effective in detecting anomalies that are close to each other
    or in datasets where anomalies are more prevalent.
    """
    features_columns = features_columns or dataset.feature_columns

    if not all(col in dataset.feature_columns for col in features_columns):
        raise ValueError(
            "The list of feature columns provided do not match with "
            + "training dataset feature columns"
        )

    df = dataset.df

    clf = IsolationForest(
        random_state=random_state,
        contamination=contamination,
    )
    clf.fit(df)
    y_pred = clf.predict(df)

    figures = []
    for feature1, feature2 in itertools.combinations(features_columns, 2):
        fig = plt.figure()

        ax = sns.scatterplot(
            data=df, x=feature1, y=feature2, hue=y_pred, palette="bright"
        )

        handles, labels = ax.get_legend_handles_labels()
        labels = list(map(lambda x: x.replace("-1", "Outliers"), labels))
        labels = list(map(lambda x: x.replace("1", "Inliers"), labels))
        ax.legend(handles, labels)

        figures.append(fig)

    return tuple(figures)
