# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from collections import defaultdict
from operator import add
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn import metrics

from validmind import tags, tasks
from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)

DEFAULT_DECAY_THRESHOLD = 0.05
DEFAULT_STD_DEV_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_CLASSIFICATION_METRIC = "auc"
DEFAULT_REGRESSION_METRIC = "mse"
PERFORMANCE_METRICS = {
    "accuracy": {
        "function": metrics.accuracy_score,
        "is_lower_better": False,
    },
    "auc": {
        "function": metrics.roc_auc_score,
        "is_lower_better": False,
    },
    "f1": {
        "function": metrics.f1_score,
        "is_lower_better": False,
    },
    "precision": {
        "function": metrics.precision_score,
        "is_lower_better": False,
    },
    "recall": {
        "function": metrics.recall_score,
        "is_lower_better": False,
    },
    "mse": {
        "function": metrics.mean_squared_error,
        "is_lower_better": True,
    },
    "mae": {
        "function": metrics.mean_absolute_error,
        "is_lower_better": True,
    },
    "r2": {
        "function": metrics.r2_score,
        "is_lower_better": False,
    },
    "mape": {
        "function": metrics.mean_absolute_percentage_error,
        "is_lower_better": True,
    },
}


def _add_noise_std_dev(
    values: List[float], x_std_dev: float
) -> Tuple[List[float], float]:
    """
    Adds Gaussian noise to a list of values.
    Args:
        values (list[float]): A list of numerical values to which noise is added.
        x_std_dev (float): A scaling factor for the standard deviation of the noise.
    Returns:
        tuple[list[float], float]: A tuple containing:
            - A list of noisy values, where each value is the sum of the corresponding value
            in the input list and a randomly generated value sampled from a Gaussian distribution
            with mean 0 and standard deviation x_std_dev times the standard deviation of the input list.
            - The standard deviation of the input list of values.
    """
    std_dev = np.std(values)
    noise_list = np.random.normal(0, x_std_dev * std_dev, size=len(values))
    noisy_values = list(map(add, noise_list, values))

    return noisy_values


def _compute_metric(
    dataset: VMDataset, model: VMModel, X: pd.DataFrame, metric: str
) -> float:
    if metric not in PERFORMANCE_METRICS:
        raise ValueError(
            f"Invalid metric: {metric}, expected one of {PERFORMANCE_METRICS.keys()}"
        )

    if metric == "auc":
        try:
            y_proba = model.predict_proba(X)
        except MissingOrInvalidModelPredictFnError:
            y_proba = model.predict(X)
        return metrics.roc_auc_score(dataset.y, y_proba)

    return PERFORMANCE_METRICS[metric]["function"](dataset.y, model.predict(X))


def _compute_gap(result: dict, metric: str) -> float:
    if PERFORMANCE_METRICS[metric]["is_lower_better"]:
        return result[metric.upper()][-1] - result[metric.upper()][0]

    return result[metric.upper()][0] - result[metric.upper()][-1]


def _combine_results(results: List[dict]):
    final_results = defaultdict(list)

    # Interleave rows from each dictionary
    for i in range(len(results[0]["Perturbation Size"])):
        for result in results:
            for key in result.keys():
                final_results[key].append(result[key][i])

    return pd.DataFrame(final_results)


def _plot_robustness(
    results: pd.DataFrame, metric: str, threshold: float, columns: List[str], model: str
):
    fig = go.Figure()

    datasets = results["Dataset"].unique()
    pallete = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in sns.color_palette("husl", len(datasets))
    ]

    for i, dataset in enumerate(datasets):
        dataset_results = results[results["Dataset"] == dataset]
        fig.add_trace(
            go.Scatter(
                x=dataset_results["Perturbation Size"],
                y=dataset_results[metric.upper()],
                mode="lines+markers",
                name=dataset,
                line=dict(width=3, color=pallete[i]),
                marker=dict(size=10),
            )
        )

    if PERFORMANCE_METRICS[metric]["is_lower_better"]:
        y_label = f"{metric.upper()} (lower is better)"
    else:
        threshold = -threshold
        y_label = f"{metric.upper()} (higher is better)"

    # add threshold lines
    for i, dataset in enumerate(datasets):
        baseline = results[results["Dataset"] == dataset][metric.upper()].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=results["Perturbation Size"].unique(),
                y=[baseline + threshold] * len(results["Perturbation Size"].unique()),
                mode="lines",
                name=f"threshold_{dataset}",
                line=dict(dash="dash", width=2, color=pallete[i]),
                showlegend=True,
            )
        )

    columns_lines = [""]
    for column in columns:
        # keep adding to the last line in list until character limit (40)
        if len(columns_lines[-1]) + len(column) < 40:
            columns_lines[-1] += f"{column}, "
        else:
            columns_lines.append(f"{column}, ")

    fig.update_layout(
        title=dict(
            text=(
                f"Model Robustness for '{model}'<br><sup>As determined by calculating "
                f"{metric.upper()} decay in the presence of random gaussian noise</sup>"
            ),
            font=dict(size=20),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=dict(
            text="Perturbation Size (X * Standard Deviation)",
        ),
        yaxis_title=dict(text=y_label),
        plot_bgcolor="white",
        margin=dict(t=60, b=80, r=20, l=60),
        xaxis=dict(showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        annotations=[
            go.layout.Annotation(
                text=f"Perturbed Features:<br><sup>{'<br>'.join(columns_lines)}</sup>",
                align="left",
                font=dict(size=14),
                bordercolor="lightgrey",
                borderwidth=1,
                borderpad=4,
                showarrow=False,
                x=1.025,
                xref="paper",
                xanchor="left",
                y=-0.15,
                yref="paper",
            )
        ],
    )

    return fig


@tags("sklearn", "model_diagnosis", "visualization")
@tasks("classification", "regression")
def RobustnessDiagnosis(
    datasets: List[VMDataset],
    model: VMModel,
    metric: str = None,
    scaling_factor_std_dev_list: List[float] = DEFAULT_STD_DEV_LIST,
    performance_decay_threshold: float = DEFAULT_DECAY_THRESHOLD,
) -> Tuple[pd.DataFrame, go.Figure, bool]:
    """
    Assesses the robustness of a machine learning model by evaluating performance decay under noisy conditions.

    ### Purpose

    The Robustness Diagnosis test aims to evaluate the resilience of a machine learning model when subjected to
    perturbations or noise in its input data. This is essential for understanding the model's ability to handle
    real-world scenarios where data may be imperfect or corrupted.

    ### Test Mechanism

    This test introduces Gaussian noise to the numeric input features of the datasets at varying scales of standard
    deviation. The performance of the model is then measured using a specified metric. The process includes:

    - Adding Gaussian noise to numerical input features based on scaling factors.
    - Evaluating the model's performance on the perturbed data using metrics like AUC for classification tasks and MSE
    for regression tasks.
    - Aggregating and plotting the results to visualize performance decay relative to perturbation size.

    ### Signs of High Risk

    - A significant drop in performance metrics with minimal noise.
    - Performance decay values exceeding the specified threshold.
    - Consistent failure to meet performance standards across multiple perturbation scales.

    ### Strengths

    - Provides insights into the model's robustness against noisy or corrupted data.
    - Utilizes a variety of performance metrics suitable for both classification and regression tasks.
    - Visualization helps in understanding the extent of performance degradation.

    ### Limitations

    - Gaussian noise might not adequately represent all types of real-world data perturbations.
    - Performance thresholds are somewhat arbitrary and might need tuning.
    - The test may not account for more complex or unstructured noise patterns that could affect model robustness.
    """
    # TODO: use single dataset
    if not metric:
        metric = (
            DEFAULT_CLASSIFICATION_METRIC
            if datasets[0].probability_column(model)
            else DEFAULT_REGRESSION_METRIC
        )

    results = [{} for _ in range(len(datasets))]

    # add baseline results (no perturbation)
    for dataset, result in zip(datasets, results):
        result["Perturbation Size"] = [0.0]
        result["Dataset"] = [f"{dataset.input_id}"]
        result["Row Count"] = [dataset._df.shape[0]]

        result[metric.upper()] = [
            _compute_metric(
                dataset=dataset,
                model=model,
                X=dataset.x_df(),
                metric=metric,
            )
        ]
        result["Performance Decay"] = [0.0]
        result["Passed"] = [True]

    # Iterate scaling factor for the standard deviation list
    for x_std_dev in scaling_factor_std_dev_list:
        for dataset, result in zip(datasets, results):

            result["Perturbation Size"].append(x_std_dev)
            result["Dataset"].append(result["Dataset"][0])
            result["Row Count"].append(result["Row Count"][0])

            temp_df = dataset.x_df().copy()
            for feature in dataset.feature_columns_numeric:
                temp_df[feature] = _add_noise_std_dev(
                    values=temp_df[feature].to_list(),
                    x_std_dev=x_std_dev,
                )

            result[metric.upper()].append(
                _compute_metric(
                    dataset=dataset,
                    model=model,
                    X=temp_df,
                    metric=metric,
                )
            )
            result["Performance Decay"].append(_compute_gap(result, metric))
            result["Passed"].append(
                result["Performance Decay"][-1] < performance_decay_threshold
            )

    results_df = _combine_results(results)
    fig = _plot_robustness(
        results=results_df,
        metric=metric,
        threshold=performance_decay_threshold,
        columns=datasets[0].feature_columns_numeric,
        model=model.input_id,
    )
    # rename perturbation size for baseline
    # Convert to object type first to avoid dtype incompatibility warning
    results_df["Perturbation Size"] = results_df["Perturbation Size"].astype(object)
    results_df.loc[
        results_df["Perturbation Size"] == 0.0, "Perturbation Size"
    ] = "Baseline (0.0)"

    return (
        results_df,
        fig,
        all(results_df["Passed"]),
    )
