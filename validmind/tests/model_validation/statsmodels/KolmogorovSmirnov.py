# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from statsmodels.stats.diagnostic import kstest_normal

from validmind import RawData, tags, tasks
from validmind.errors import InvalidTestParametersError
from validmind.vm_models import VMDataset, VMModel


@tags("tabular_data", "data_distribution", "statistical_test", "statsmodels")
@tasks("classification", "regression")
def KolmogorovSmirnov(
    model: VMModel, dataset: VMDataset, dist: str = "norm"
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Assesses whether each feature in the dataset aligns with a normal distribution using the Kolmogorov-Smirnov test.

    ### Purpose

    The Kolmogorov-Smirnov (KS) test evaluates the distribution of features in a dataset to determine their alignment
    with a normal distribution. This is important because many statistical methods and machine learning models assume
    normality in the data distribution.

    ### Test Mechanism

    This test calculates the KS statistic and corresponding p-value for each feature in the dataset. It does so by
    comparing the cumulative distribution function of the feature with an ideal normal distribution. The KS statistic
    and p-value for each feature are then stored in a dictionary. The p-value threshold to reject the normal
    distribution hypothesis is not preset, providing flexibility for different applications.

    ### Signs of High Risk

    - Elevated KS statistic for a feature combined with a low p-value, indicating a significant divergence from a
    normal distribution.
    - Features with notable deviations that could create problems if the model assumes normality in data distribution.

    ### Strengths

    - The KS test is sensitive to differences in the location and shape of empirical cumulative distribution functions.
    - It is non-parametric and adaptable to various datasets, as it does not assume any specific data distribution.
    - Provides detailed insights into the distribution of individual features.

    ### Limitations

    - The test's sensitivity to disparities in the tails of data distribution might cause false alarms about
    non-normality.
    - Less effective for multivariate distributions, as it is designed for univariate distributions.
    - Does not identify specific types of non-normality, such as skewness or kurtosis, which could impact model fitting.
    """
    if dist not in ["norm", "exp"]:
        raise InvalidTestParametersError(
            "'dist' parameter must be either 'norm' or 'exp'"
        )

    df = dataset.df[dataset.feature_columns_numeric]

    ks_values = {}
    for col in df.columns:
        ks_stat, p_value = kstest_normal(df[col].values, dist)
        ks_values[col] = {"stat": ks_stat, "pvalue": p_value}

    return [
        {
            "Column": k,
            "Statistic": result["stat"],
            "P-Value": result["pvalue"],
        }
        for k, result in ks_values.items()
    ], RawData(ks_values=ks_values, dataset=dataset.input_id)
