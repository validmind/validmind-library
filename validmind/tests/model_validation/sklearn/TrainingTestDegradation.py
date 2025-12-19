# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from numpy import unique
from sklearn.metrics import classification_report

from validmind import RawData
from validmind.tests import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def TrainingTestDegradation(
    datasets: List[VMDataset], model: VMModel, max_threshold: float = 0.10
) -> Tuple[List[Dict[str, float]], bool, RawData]:
    """
    Tests if model performance degradation between training and test datasets exceeds a predefined threshold.

    ### Purpose

    The `TrainingTestDegradation` class serves as a test to verify that the degradation in performance between the
    training and test datasets does not exceed a predefined threshold. This test measures the model's ability to
    generalize from its training data to unseen test data, assessing key classification metrics such as precision,
    recall, and f1 score to verify the model's robustness and reliability.

    ### Test Mechanism

    The code applies several predefined metrics, including precision, recall, and f1 scores, to the model's
    predictions for both the training and test datasets. It calculates the degradation as the difference between the
    training score and test score divided by the training score. The test is considered successful if the degradation
    for each metric is less than the preset maximum threshold (default: 0.10). The results are summarized in a table
    showing each metric's train score, test score, degradation percentage, and pass/fail status.

    ### Signs of High Risk

    - A degradation percentage that exceeds the maximum allowed threshold of 10% for any of the evaluated metrics.
    - A high difference or gap between the metric scores on the training and the test datasets.
    - The 'Pass/Fail' column displaying 'Fail' for any of the evaluated metrics.

    ### Strengths

    - Provides a quantitative measure of the model's ability to generalize to unseen data, which is key for predicting
    its practical real-world performance.
    - By evaluating multiple metrics, it takes into account different facets of model performance and enables a more
    holistic evaluation.
    - The use of a variable predefined threshold allows the flexibility to adjust the acceptability criteria for
    different scenarios.

    ### Limitations

    - The test compares raw performance on training and test data but does not factor in the nature of the data. Areas
    with less representation in the training set might still perform poorly on unseen data.
    - It requires good coverage and balance in the test and training datasets to produce reliable results, which may
    not always be available.
    - The test is currently only designed for classification tasks.
    """
    ds1_report = classification_report(
        y_true=datasets[0].y,
        y_pred=datasets[0].y_pred(model),
        output_dict=True,
        zero_division=0,
    )
    ds2_report = classification_report(
        y_true=datasets[1].y,
        y_pred=datasets[1].y_pred(model),
        output_dict=True,
        zero_division=0,
    )

    table = []

    for class_name in {str(i) for i in unique(datasets[0].y)}:
        for metric_name in ["precision", "recall", "f1-score"]:
            ds1_score = ds1_report[class_name][metric_name]
            ds2_score = ds2_report[class_name][metric_name]

            # If training score is 0, degradation is assumed to be 100%
            degradation = 1.0 if ds1_score == 0 else (ds1_score - ds2_score) / ds1_score
            passed = degradation < max_threshold

            table.append(
                {
                    "Class": class_name,
                    "Metric": metric_name.title(),
                    f"{datasets[0].input_id} Score": ds1_score,
                    f"{datasets[1].input_id} Score": ds2_score,
                    "Degradation (%)": degradation * 100,
                    "Pass/Fail": "Pass" if passed else "Fail",
                }
            )

    return (
        table,
        all(row["Pass/Fail"] == "Pass" for row in table),
        RawData(
            dataset_1_report=ds1_report,
            dataset_2_report=ds2_report,
        ),
    )
