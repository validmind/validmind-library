# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMModel

logger = get_logger(__name__)


@tags("statistical_test", "model_interpretation", "visualization", "feature_importance")
@tasks("regression")
def RegressionFeatureSignificance(
    model: VMModel,
    fontsize: int = 10,
    p_threshold: float = 0.05,
) -> Tuple[go.Figure, RawData]:
    """
    Assesses and visualizes the statistical significance of features in a regression model.

    ### Purpose

    The Regression Feature Significance metric assesses the significance of each feature in a given set of regression
    model. It creates a visualization displaying p-values for every feature of the model, assisting model developers
    in understanding which features are most influential in their model.

    ### Test Mechanism

    The test mechanism involves extracting the model's coefficients and p-values for each feature, and then plotting these
    values. The x-axis on the plot contains the p-values while the y-axis denotes the coefficients of each feature. A
    vertical red line is drawn at the threshold for p-value significance, which is 0.05 by default. Any features with
    p-values to the left of this line are considered statistically significant at the chosen level.

    ### Signs of High Risk

    - Any feature with a high p-value (greater than the threshold) is considered a potential high risk, as it suggests
    the feature is not statistically significant and may not be reliably contributing to the model's predictions.
    - A high number of such features may indicate problems with the model validation, variable selection, and overall
    reliability of the model predictions.

    ### Strengths

    - Helps identify the features that significantly contribute to a model's prediction, providing insights into the
    feature importance.
    - Provides tangible, easy-to-understand visualizations to interpret the feature significance.

    ### Limitations

    - This metric assumes model features are independent, which may not always be the case. Multicollinearity (high
    correlation amongst predictors) can cause high variance and unreliable statistical tests of significance.
    - The p-value strategy for feature selection doesn't take into account the magnitude of the effect, focusing solely
    on whether the feature is likely non-zero.
    - This test is specific to regression models and wouldn't be suitable for other types of ML models.
    - P-value thresholds are somewhat arbitrary and do not always indicate practical significance, only statistical
    significance.
    """
    if model.library != "statsmodels":
        raise SkipTestError("Only statsmodels are supported for this metric")

    coefficients = model.model.params
    pvalues = model.model.pvalues

    # Sort the variables by p-value in ascending order
    sorted_idx = pvalues.argsort()
    coefficients = coefficients.iloc[sorted_idx]
    pvalues = pvalues.iloc[sorted_idx]

    fig, ax = plt.subplots()

    sns.barplot(x=pvalues, y=coefficients.index, ax=ax, color="skyblue")

    # Add a threshold line at p-value = p_threshold
    threshold_line = ax.axvline(x=p_threshold, color="red", linestyle="--")

    # Set labels and title
    ax.set_xlabel("P-value")
    ax.set_ylabel(None)
    ax.set_title(f"Feature Significance for {model.input_id}")

    plt.tight_layout()

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)

    # Add a legend for the threshold line
    legend_label = f"p_threshold {p_threshold}"
    ax.legend([threshold_line], [legend_label])

    plt.close()

    return fig, RawData(
        coefficients=coefficients, pvalues=pvalues, model=model.input_id
    )
