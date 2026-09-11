# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from validmind import RawData, tags, tasks
from validmind.tests.model_validation.sklearn._multiclass_proba import multiclass_proba
from validmind.vm_models import VMDataset, VMModel


@tags("model_performance")
@tasks("classification")
def GINITable(dataset: VMDataset, model: VMModel) -> Tuple[pd.DataFrame, RawData]:
    """
    Evaluates classification model performance using AUC, GINI, and KS metrics for training and test datasets.

    ### Purpose

    The 'GINITable' metric is designed to evaluate the performance of a classification model by emphasizing its
    discriminatory power. Specifically, it calculates and presents three important metrics - the Area under the ROC
    Curve (AUC), the GINI coefficient, and the Kolmogorov-Smirnov (KS) statistic - for both training and test datasets.

    ### Test Mechanism

    Using a dictionary for storing performance metrics for both the training and test datasets, the 'GINITable' metric
    calculates each of these metrics sequentially. The Area under the ROC Curve (AUC) is calculated via the
    `roc_auc_score` function from the Scikit-Learn library. The GINI coefficient, a measure of statistical dispersion,
    is then computed by doubling the AUC and subtracting 1. Finally, the Kolmogorov-Smirnov (KS) statistic is
    calculated via the `roc_curve` function from Scikit-Learn, with the False Positive Rate (FPR) subtracted from the
    True Positive Rate (TPR) and the maximum value taken from the resulting data. These metrics are then stored in a
    pandas DataFrame for convenient visualization.

    ### Signs of High Risk

    - Low values for performance metrics may suggest a reduction in model performance, particularly a low AUC which
    indicates poor classification performance, or a low GINI coefficient, which could suggest a decreased ability to
    discriminate different classes.
    - A high KS value may be an indicator of potential overfitting, as this generally signifies a substantial
    divergence between positive and negative distributions.
    - Significant discrepancies between the performance on the training dataset and the test dataset may present
    another signal of high risk.

    ### Strengths

    - Offers three key performance metrics (AUC, GINI, and KS) in one test, providing a more comprehensive evaluation
    of the model.
    - Provides a direct comparison between the model's performance on training and testing datasets, which aids in
    identifying potential underfitting or overfitting.
    - The applied metrics are class-distribution invariant, thereby remaining effective for evaluating model
    performance even when dealing with imbalanced datasets.
    - Presents the metrics in a user-friendly table format for easy comprehension and analysis.

    ### Limitations

    - The GINI coefficient and KS statistic are both dependent on the AUC value. Therefore, any errors in the
    calculation of the latter will adversely impact the former metrics too.
    - For multiclass models the metrics are computed one-vs-rest (one row per class plus a micro-average row), which
    requires per-class probabilities from the model's `predict_proba`. Models that cannot produce a full per-class
    probability matrix (e.g. metadata-only models, or predictions supplied as a single precomputed probability column)
    are skipped for the multiclass case rather than crashed.
    - The metrics used are threshold-dependent and may exhibit high variability based on the chosen cut-off points.
    - The test does not incorporate a method to efficiently handle missing or inefficiently processed data, which could
    lead to inaccuracies in the metrics if the data is not appropriately preprocessed.
    """
    classes = np.unique(dataset.y)

    if len(classes) > 2:
        return _multiclass_gini_table(model, dataset)

    y_true = np.ravel(dataset.y)  # Flatten y_true to make it one-dimensional
    y_prob = dataset.y_prob(model)
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    return pd.DataFrame(
        {
            "AUC": [auc],
            "GINI": [gini],
            "KS": [max(tpr - fpr)],
        }
    ), RawData(
        fpr=fpr,
        tpr=tpr,
        y_true=y_true,
        y_prob=y_prob,
        model=model.input_id,
        dataset=dataset.input_id,
    )


def _multiclass_gini_table(
    model: VMModel, dataset: VMDataset
) -> Tuple[pd.DataFrame, RawData]:
    """One-vs-rest AUC/GINI/KS for a multiclass model.

    Needs the full per-class probability matrix, which the stored single
    probability column cannot provide; the shared helper reaches the underlying
    estimator, aligns the probability columns to the training class order and
    skips models that cannot supply a matching matrix.
    """
    aligned = multiclass_proba(model, dataset, "GINITable")
    y_true = np.asarray(dataset.y).flatten()
    y_bin = aligned.y_bin
    y_prob = aligned.y_prob

    rows = []
    raw_fpr = {}
    raw_tpr = {}
    for i, cls in zip(aligned.present_indices, aligned.classes_present):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        key = str(cls)
        raw_fpr[key] = fpr
        raw_tpr[key] = tpr
        rows.append(
            {"Class": key, "AUC": auc, "GINI": 2 * auc - 1, "KS": max(tpr - fpr)}
        )

    # Micro-average across the one-vs-rest decisions of the present classes.
    present = aligned.present_indices
    micro_fpr, micro_tpr, _ = roc_curve(
        y_bin[:, present].ravel(), y_prob[:, present].ravel()
    )
    micro_auc = roc_auc_score(
        y_bin[:, present], y_prob[:, present], average="micro", multi_class="ovr"
    )
    raw_fpr["micro"] = micro_fpr
    raw_tpr["micro"] = micro_tpr
    rows.append(
        {
            "Class": "micro",
            "AUC": micro_auc,
            "GINI": 2 * micro_auc - 1,
            "KS": max(micro_tpr - micro_fpr),
        }
    )

    return pd.DataFrame(rows), RawData(
        fpr=raw_fpr,
        tpr=raw_tpr,
        y_true=y_true,
        y_prob=y_prob,
        model=model.input_id,
        dataset=dataset.input_id,
    )
