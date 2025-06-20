# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Any, Dict, List, Tuple

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset


@tags("tabular_data", "data_quality", "correlation")
@tasks("classification", "regression")
def HighPearsonCorrelation(
    dataset: VMDataset,
    max_threshold: float = 0.3,
    top_n_correlations: int = 10,
    feature_columns: list = None,
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Identifies highly correlated feature pairs in a dataset suggesting feature redundancy or multicollinearity.

    ### Purpose

    The High Pearson Correlation test measures the linear relationship between features in a dataset, with the main
    goal of identifying high correlations that might indicate feature redundancy or multicollinearity. Identification
    of such issues allows developers and risk management teams to properly deal with potential impacts on the machine
    learning model's performance and interpretability.

    ### Test Mechanism

    The test works by generating pairwise Pearson correlations for all features in the dataset, then sorting and
    eliminating duplicate and self-correlations. It assigns a Pass or Fail based on whether the absolute value of the
    correlation coefficient surpasses a pre-set threshold (defaulted at 0.3). It lastly returns the top n strongest
    correlations regardless of passing or failing status (where n is 10 by default but can be configured by passing the
    `top_n_correlations` parameter).

    ### Signs of High Risk

    - A high risk indication would be the presence of correlation coefficients exceeding the threshold.
    - If the features share a strong linear relationship, this could lead to potential multicollinearity and model
    overfitting.
    - Redundancy of variables can undermine the interpretability of the model due to uncertainty over the authenticity
    of individual variable's predictive power.

    ### Strengths

    - Provides a quick and simple means of identifying relationships between feature pairs.
    - Generates a transparent output that displays pairs of correlated variables, the Pearson correlation coefficient,
    and a Pass or Fail status for each.
    - Aids in early identification of potential multicollinearity issues that may disrupt model training.

    ### Limitations

    - Can only delineate linear relationships, failing to shed light on nonlinear relationships or dependencies.
    - Sensitive to outliers where a few outliers could notably affect the correlation coefficient.
    - Limited to identifying redundancy only within feature pairs; may fail to spot more complex relationships among
    three or more variables.
    """

    # Select features
    if feature_columns is None:
        df = dataset.df
    else:
        df = dataset.df[feature_columns]

    # Get correlation matrix for numeric columns
    corr = df.corr(numeric_only=True)

    # Create table of correlation coefficients and column pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            coeff = corr.iloc[i, j]
            pairs.append(
                {
                    "Columns": f"({corr.columns[i]}, {corr.columns[j]})",
                    "Coefficient": coeff,
                    "Pass/Fail": "Pass" if abs(coeff) <= max_threshold else "Fail",
                }
            )

    # Sort by absolute coefficient and get top N
    pairs.sort(key=lambda x: abs(x["Coefficient"]), reverse=True)
    pairs = pairs[:top_n_correlations]

    return (
        pairs,
        all(p["Pass/Fail"] == "Pass" for p in pairs),
        RawData(correlation_matrix=corr, dataset=dataset.input_id),
    )
