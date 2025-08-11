# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.utils import format_records
from validmind.vm_models import VMDataset


def _validate_columns(dataset: VMDataset, columns: Optional[List[str]]):
    """Validate and return numerical columns."""
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for normality testing")

    return columns


def _run_shapiro_test(data, tests: List[str], alpha: float):
    """Run Shapiro-Wilk test if requested and data size is appropriate."""
    results = {}
    if "shapiro" in tests and len(data) <= 5000:
        try:
            stat, p_value = stats.shapiro(data)
            results["Shapiro-Wilk Stat"] = stat
            results["Shapiro-Wilk p-value"] = p_value
            results["Shapiro-Wilk Normal"] = "Yes" if p_value > alpha else "No"
        except Exception:
            results["Shapiro-Wilk Normal"] = "Test Failed"
    return results


def _run_anderson_test(data, tests: List[str]):
    """Run Anderson-Darling test if requested."""
    results = {}
    if "anderson" in tests:
        try:
            ad_result = stats.anderson(data, dist="norm")
            critical_value = ad_result.critical_values[2]  # 5% level
            results["Anderson-Darling Stat"] = ad_result.statistic
            results["Anderson-Darling Critical"] = critical_value
            results["Anderson-Darling Normal"] = (
                "Yes" if ad_result.statistic < critical_value else "No"
            )
        except Exception:
            results["Anderson-Darling Normal"] = "Test Failed"
    return results


def _run_ks_test(data, tests: List[str], alpha: float):
    """Run Kolmogorov-Smirnov test if requested."""
    results = {}
    if "kstest" in tests:
        try:
            standardized = (data - data.mean()) / data.std()
            stat, p_value = stats.kstest(standardized, "norm")
            results["KS Test Stat"] = stat
            results["KS Test p-value"] = p_value
            results["KS Test Normal"] = "Yes" if p_value > alpha else "No"
        except Exception:
            results["KS Test Normal"] = "Test Failed"
    return results


def _process_column_tests(column: str, data, tests: List[str], alpha: float):
    """Process all normality tests for a single column."""
    result_row = {"Feature": column, "Sample Size": len(data)}

    # Run individual tests
    result_row.update(_run_shapiro_test(data, tests, alpha))
    result_row.update(_run_anderson_test(data, tests))
    result_row.update(_run_ks_test(data, tests, alpha))

    return result_row


@tags("tabular_data", "statistics", "normality")
@tasks("classification", "regression", "clustering")
def NormalityTests(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    tests: List[str] = ["shapiro", "anderson", "kstest"],
) -> Dict[str, Any]:
    """
    Performs multiple normality tests on numerical features to assess distribution normality.

    ### Purpose

    This test evaluates whether numerical features follow a normal distribution using
    various statistical tests. Understanding distribution normality is crucial for
    selecting appropriate statistical methods and model assumptions.

    ### Test Mechanism

    The test applies multiple normality tests:
    - Shapiro-Wilk test: Best for small to medium samples
    - Anderson-Darling test: More sensitive to deviations in tails
    - Kolmogorov-Smirnov test: General goodness-of-fit test

    ### Signs of High Risk

    - Multiple normality tests failing consistently
    - Very low p-values indicating strong evidence against normality
    - Conflicting results between different normality tests

    ### Strengths

    - Multiple statistical tests for robust assessment
    - Clear pass/fail indicators for each test
    - Suitable for different sample sizes

    ### Limitations

    - Limited to numerical features only
    - Some tests sensitive to sample size
    - Perfect normality is rare in real data
    """
    # Validate inputs
    columns = _validate_columns(dataset, columns)

    # Process each column
    normality_results = []
    for column in columns:
        data = dataset.df[column].dropna()

        if len(data) >= 3:
            result_row = _process_column_tests(column, data, tests, alpha)
            normality_results.append(result_row)

    # Format results
    results = {}
    if normality_results:
        results["Normality Tests"] = format_records(pd.DataFrame(normality_results))

    return results
