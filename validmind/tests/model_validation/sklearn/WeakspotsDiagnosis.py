# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

from validmind.vm_models import (
    Figure,
    ResultSummary,
    ResultTable,
    ResultTableMetadata,
    ThresholdTest,
    ThresholdTestResult,
)


@dataclass
class WeakspotsDiagnosis(ThresholdTest):
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

    name = "weak_spots"
    required_inputs = ["model", "datasets"]

    default_params = {
        "features_columns": None,
        # Some default values that the user should override
        "thresholds": {
            "accuracy": 0.75,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.7,
        },
    }

    tasks = ["classification", "text_classification"]
    tags = [
        "sklearn",
        "binary_classification",
        "multiclass_classification",
        "model_diagnosis",
        "visualization",
    ]

    # TODO: allow configuring
    default_metrics = {
        "accuracy": metrics.accuracy_score,
        "precision": partial(metrics.precision_score, zero_division=0),
        "recall": partial(metrics.recall_score, zero_division=0),
        "f1": partial(metrics.f1_score, zero_division=0),
    }

    def run(self):
        thresholds = self.params["thresholds"]

        # Ensure there is a threshold for each metric
        for metric in self.default_metrics.keys():
            if metric not in thresholds:
                raise ValueError(f"Threshold for metric {metric} is missing")

        if self.params["features_columns"] is None:
            features_list = self.inputs.datasets[0].feature_columns
        else:
            features_list = self.params["features_columns"]

        if self.inputs.datasets[0].text_column in features_list:
            raise ValueError(
                "Skiping Weakspots Diagnosis test for the dataset with text column"
            )

        # Check if all elements from features_list are present in the feature columns
        all_present = all(
            elem in self.inputs.datasets[0].feature_columns for elem in features_list
        )
        if not all_present:
            raise ValueError(
                "The list of feature columns provided do not match with "
                + "training dataset feature columns"
            )

        target_column = self.inputs.datasets[0].target_column
        prediction_column = f"{target_column}_pred"

        train_df = self.inputs.datasets[0].df.copy()
        train_class_pred = self.inputs.datasets[0].y_pred(self.inputs.model)
        train_df[prediction_column] = train_class_pred

        test_df = self.inputs.datasets[1].df.copy()
        test_class_pred = self.inputs.datasets[1].y_pred(self.inputs.model)
        test_df[prediction_column] = test_class_pred

        test_results = []
        test_figures = []
        results_headers = ["slice", "shape", "feature"]
        results_headers.extend(self.default_metrics.keys())
        for feature in features_list:
            bins = 10
            if feature in self.inputs.datasets[0].feature_columns_categorical:
                bins = len(train_df[feature].unique())
            train_df["bin"] = pd.cut(train_df[feature], bins=bins)

            results_train = {k: [] for k in results_headers}
            results_test = {k: [] for k in results_headers}

            for region, df_region in train_df.groupby("bin"):
                self._compute_metrics(
                    results_train,
                    region,
                    df_region,
                    target_column,
                    prediction_column,
                    feature,
                )
                df_test_region = test_df[
                    (test_df[feature] > region.left)
                    & (test_df[feature] <= region.right)
                ]
                self._compute_metrics(
                    results_test,
                    region,
                    df_test_region,
                    target_column,
                    prediction_column,
                    feature,
                )

            # Make one plot per metric
            for metric in self.default_metrics.keys():
                fig, df = self._plot_weak_spots(
                    results_train,
                    results_test,
                    feature,
                    metric=metric,
                    threshold=thresholds[metric],
                )

                test_figures.append(
                    Figure(
                        for_object=self,
                        key=f"{self.name}:{metric}:{feature}",
                        figure=fig,
                        metadata={
                            "metric": metric,
                            "threshold": thresholds[metric],
                            "feature": feature,
                        },
                    )
                )

            # For simplicity, test has failed if any of the metrics is below the threshold. We will
            # rely on visual assessment for this test for now.
            results_passed = df[df[list(thresholds.keys())].lt(thresholds).any(axis=1)]
            passed = results_passed.empty

            test_results.append(
                ThresholdTestResult(
                    test_name="accuracy",
                    column=feature,
                    passed=passed,
                    values={"records": df.to_dict("records")},
                )
            )
        return self.cache_results(
            test_results,
            passed=all([r.passed for r in test_results]),
            figures=test_figures,
        )

    def summary(self, results: List[ThresholdTestResult], all_passed: bool):
        results_table = [
            record for result in results for record in result.values["records"]
        ]
        return ResultSummary(
            results=[
                ResultTable(
                    data=results_table,
                    metadata=ResultTableMetadata(title="Weakspots Test"),
                )
            ]
        )

    def _compute_metrics(
        self,
        results: dict,
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
        results["slice"].append(str(region))
        results["shape"].append(df_region.shape[0])
        results["feature"].append(feature_column)

        # Check if df_region is an empty dataframe and if so, append 0 to all metrics
        if df_region.empty:
            for metric in self.default_metrics.keys():
                results[metric].append(0)
            return

        y_true = df_region[target_column].values
        y_prediction = (
            df_region[prediction_column].astype(df_region[target_column].dtypes).values
        )

        for metric, metric_fn in self.default_metrics.items():
            results[metric].append(metric_fn(y_true, y_prediction))

    def _plot_weak_spots(
        self, results_train, results_test, feature_column, metric, threshold
    ):
        """
        Plots the metric of the training and test datasets for each region in a given feature column,
        and highlights regions where the score is below a specified threshold.
        Args:
            results_train (list of dict): The results of the model on the training dataset, as a list of dictionaries.
            results_test (list of dict): The results of the model on the test dataset, as a list of dictionaries.
            feature_column (str): The name of the feature column being analyzed.
            metric (str): The name of the metric to plot.
            threshold (float): The minimum accuracy threshold to be highlighted on the plot.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            df (pandas.DataFrame): The concatenated dataframe containing the training and test results.
        """
        # Concat training and test datasets
        results_train = pd.DataFrame(results_train)
        results_test = pd.DataFrame(results_test)
        dataset_type_column = "Dataset Type"
        results_train[dataset_type_column] = "Training"
        results_test[dataset_type_column] = "Test"
        df = pd.concat([results_train, results_test])

        # Create a bar plot using seaborn library
        fig, ax = plt.subplots()
        barplot = sns.barplot(
            data=df,
            x="slice",
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

        # Do this if you want to prevent the figure from being displayed
        plt.close("all")

        return fig, df
