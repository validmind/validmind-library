# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn import metrics

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

DEFAULT_METRICS = {
    "accuracy": metrics.accuracy_score,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score,
    "f1": metrics.f1_score,
}
DEFAULT_THRESHOLDS = {
    "accuracy": 0.75,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.7,
}


def _compute_metrics(
    results: dict,
    metrics: Dict[str, Callable],
    region: str,
    df_region: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    feature_column: str,
) -> None:
    """
    Computes and appends the default metrics for a given DataFrame slice to the results dictionary.
    Args:
        results (dict): A dictionary to which the computed metrics will be appended.
        region (str): A string identifier for the DataFrame slice being evaluated.
        df_region (pd.DataFrame): A pandas DataFrame slice containing the data to evaluate.
        target_column (str): The name of the target column to use for computing the metrics.
        prediction_column (str): The name of the prediction column to use for computing the metrics.
    Returns:
        None: The computed metrics are appended to the `results` dictionary in-place.
    """
    results["Slice"].append(str(region))
    results["Number of Records"].append(df_region.shape[0])
    results["Feature"].append(feature_column)

    # Check if df_region is an empty dataframe and if so, append 0 to all metrics
    if df_region.empty:
        for metric in metrics.keys():
            results[metric].append(0)
        return

    y_true = df_region[target_column].values
    y_prediction = (
        df_region[prediction_column].astype(df_region[target_column].dtypes).values
    )

    for metric, metric_fn in metrics.items():
        results[metric].append(metric_fn(y_true, y_prediction))


def _plot_weak_spots(
    results_1: dict, results_2: dict, feature_column: str, metric: str, threshold: float
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plots the metric of the training and test datasets for each region in a given feature column,
    and highlights regions where the score is below a specified threshold.
    Args:
        results_1 (list of dict): The results of the model on the training dataset, as a list of dictionaries.
        results_2 (list of dict): The results of the model on the test dataset, as a list of dictionaries.
        feature_column (str): The name of the feature column being analyzed.
        metric (str): The name of the metric to plot.
        threshold (float): The minimum accuracy threshold to be highlighted on the plot.
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        df (pandas.DataFrame): The concatenated dataframe containing the training and test results.
    """
    # Concat training and test datasets
    results_1 = pd.DataFrame(results_1)
    results_2 = pd.DataFrame(results_2)
    dataset_type_column = "Dataset Type"
    results_1[dataset_type_column] = "Training"
    results_2[dataset_type_column] = "Test"
    df = pd.concat([results_1, results_2])

    # Create a bar plot using seaborn library
    fig, ax = plt.subplots()
    barplot = sns.barplot(
        data=df,
        x="Slice",
        y=metric,
        hue=dataset_type_column,
        edgecolor="black",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    for p in ax.patches:
        t = ax.annotate(
            str("{:.2f}%".format(p.get_height())),
            xy=(p.get_x() + 0.03, p.get_height() + 1),
        )
        t.set(color="black", size=14)

    axhline = ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Threshold: {threshold}",
    )
    ax.set_ylabel(metric.capitalize(), weight="bold", fontsize=18)
    ax.set_xlabel("Slice/Segments", weight="bold", fontsize=18)
    ax.set_title(
        f"Weak regions in feature column: {feature_column}",
        weight="bold",
        fontsize=20,
        wrap=True,
    )

    # Get the legend handles and labels from the barplot
    handles, labels = barplot.get_legend_handles_labels()

    # Append the axhline handle and label
    handles.append(axhline)
    labels.append(axhline.get_label())

    # Create a legend with both hue and axhline labels, the threshold line
    # will show up twice so remove the first element
    # barplot.legend(handles=handles[:-1], labels=labels, loc="upper right")
    barplot.legend(
        handles=handles[:-1],
        labels=labels[:-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(handles) - 1,
    )

    plt.close()

    return fig, df


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_diagnosis",
    "visualization",
)
@tasks("classification", "text_classification")
def WeakspotsDiagnosis(
    datasets: List[VMDataset],
    model: VMModel,
    features_columns: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, go.Figure, bool]:
    """
    Identifies and visualizes weak spots in a machine learning model's performance across various sections of the
    feature space.

    ### Purpose

    The weak spots test is applied to evaluate the performance of a machine learning model within specific regions of
    its feature space. This test slices the feature space into various sections, evaluating the model's outputs within
    each section against specific performance metrics (e.g., accuracy, precision, recall, and F1 scores). The ultimate
    aim is to identify areas where the model's performance falls below the set thresholds, thereby exposing its
    possible weaknesses and limitations.

    ### Test Mechanism

    The test mechanism adopts an approach of dividing the feature space of the training dataset into numerous bins. The
    model's performance metrics (accuracy, precision, recall, F1 scores) are then computed for each bin on both the
    training and test datasets. A "weak spot" is identified if any of the performance metrics fall below a
    predetermined threshold for a particular bin on the test dataset. The test results are visually plotted as bar
    charts for each performance metric, indicating the bins which fail to meet the established threshold.

    ### Signs of High Risk

    - Any performance metric of the model dropping below the set thresholds.
    - Significant disparity in performance between the training and test datasets within a bin could be an indication
    of overfitting.
    - Regions or slices with consistently low performance metrics. Such instances could mean that the model struggles
    to handle specific types of input data adequately, resulting in potentially inaccurate predictions.

    ### Strengths

    - The test helps pinpoint precise regions of the feature space where the model's performance is below par, allowing
    for more targeted improvements to the model.
    - The graphical presentation of the performance metrics offers an intuitive way to understand the model's
    performance across different feature areas.
    - The test exhibits flexibility, letting users set different thresholds for various performance metrics according
    to the specific requirements of the application.

    ### Limitations

    - The binning system utilized for the feature space in the test could over-simplify the model's behavior within
    each bin. The granularity of this slicing depends on the chosen 'bins' parameter and can sometimes be arbitrary.
    - The effectiveness of this test largely hinges on the selection of thresholds for the performance metrics, which
    may not hold universally applicable and could be subjected to the specifications of a particular model and
    application.
    - The test is unable to handle datasets with a text column, limiting its application to numerical or categorical
    data types only.
    - Despite its usefulness in highlighting problematic regions, the test does not offer direct suggestions for model
    improvement.
    """
    feature_columns = features_columns or datasets[0].feature_columns
    numeric_and_categorical_columns = (
        datasets[0].feature_columns_numeric + datasets[0].feature_columns_categorical
    )

    feature_columns = [
        col for col in feature_columns if col in numeric_and_categorical_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No valid numeric or categorical columns found in features_columns"
        )

    if not all(col in datasets[0].feature_columns for col in feature_columns):
        raise ValueError(
            "Column(s) provided in features_columns do not exist in the dataset"
        )

    metrics = metrics or DEFAULT_METRICS
    metrics = {k.title(): v for k, v in metrics.items()}

    thresholds = thresholds or DEFAULT_THRESHOLDS
    thresholds = {k.title(): v for k, v in thresholds.items()}

    results_headers = ["Slice", "Number of Records", "Feature"]
    results_headers.extend(metrics.keys())

    figures = []
    passed = True

    df_1 = datasets[0]._df[
        feature_columns
        + [datasets[0].target_column, datasets[0].prediction_column(model)]
    ]
    df_2 = datasets[1]._df[
        feature_columns
        + [datasets[1].target_column, datasets[1].prediction_column(model)]
    ]
    results_1 = pd.DataFrame()
    results_2 = pd.DataFrame()
    for feature in feature_columns:
        bins = 10
        if feature in datasets[0].feature_columns_categorical:
            bins = len(df_1[feature].unique())
        df_1["bin"] = pd.cut(df_1[feature], bins=bins)

        r1 = {k: [] for k in results_headers}
        r2 = {k: [] for k in results_headers}

        for region, df_region in df_1.groupby("bin", observed=True):
            _compute_metrics(
                results=r1,
                metrics=metrics,
                region=region,
                df_region=df_region,
                target_column=datasets[0].target_column,
                prediction_column=datasets[0].prediction_column(model),
                feature_column=feature,
            )
            df_2_region = df_2[
                (df_2[feature] > region.left) & (df_2[feature] <= region.right)
            ]
            _compute_metrics(
                results=r2,
                metrics=metrics,
                region=region,
                df_region=df_2_region,
                target_column=datasets[1].target_column,
                prediction_column=datasets[1].prediction_column(model),
                feature_column=feature,
            )

        for metric in metrics.keys():
            fig, df = _plot_weak_spots(
                results_1=r1,
                results_2=r2,
                feature_column=feature,
                metric=metric,
                threshold=thresholds[metric],
            )

            figures.append(fig)

        # For simplicity, test has failed if any of the metrics is below the threshold. We will
        # rely on visual assessment for this test for now.
        if not df[df[list(thresholds.keys())].lt(thresholds).any(axis=1)].empty:
            passed = False
        results_1 = pd.concat([results_1, pd.DataFrame(r1)])
        results_2 = pd.concat([results_2, pd.DataFrame(r2)])

    return (
        pd.concat(
            [
                pd.DataFrame(results_1).assign(Dataset=datasets[0].input_id),
                pd.DataFrame(results_2).assign(Dataset=datasets[1].input_id),
            ]
        )
        .reset_index(drop=True)
        .sort_values(["Feature", "Dataset"]),
        *figures,
        passed,
    )
