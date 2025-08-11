# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.utils import format_records
from validmind.vm_models import VMDataset


def _validate_and_prepare_data(dataset: VMDataset, columns: Optional[List[str]]):
    """Validate inputs and prepare data for correlation analysis."""
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for correlation analysis")

    if len(columns) < 2:
        raise SkipTestError(
            "At least 2 numerical columns required for correlation analysis"
        )

    # Get data and remove constant columns
    data = dataset.df[columns].dropna()
    data = data.loc[:, data.var() != 0]

    if data.shape[1] < 2:
        raise SkipTestError(
            "Insufficient non-constant columns for correlation analysis"
        )

    return data


def _compute_correlation_matrices(data, method: str):
    """Compute correlation and p-value matrices based on method."""
    if method == "pearson":
        return _compute_pearson_with_pvalues(data)
    elif method == "spearman":
        return _compute_spearman_with_pvalues(data)
    elif method == "kendall":
        return _compute_kendall_with_pvalues(data)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")


def _create_correlation_pairs(
    corr_matrix, p_matrix, significance_level: float, min_correlation: float
):
    """Create correlation pairs table."""
    correlation_pairs = []

    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_matrix.iloc[i, j]

                if abs(corr_val) >= min_correlation:
                    pair_info = {
                        "Feature 1": col1,
                        "Feature 2": col2,
                        "Correlation": corr_val,
                        "Abs Correlation": abs(corr_val),
                        "p-value": p_val,
                        "Significant": "Yes" if p_val < significance_level else "No",
                        "Strength": _correlation_strength(abs(corr_val)),
                        "Direction": "Positive" if corr_val > 0 else "Negative",
                    }
                    correlation_pairs.append(pair_info)

    # Sort by absolute correlation value
    correlation_pairs.sort(key=lambda x: x["Abs Correlation"], reverse=True)
    return correlation_pairs


def _create_summary_statistics(corr_matrix, correlation_pairs):
    """Create summary statistics table."""
    all_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            all_correlations.append(abs(corr_matrix.iloc[i, j]))

    significant_count = sum(
        1 for pair in correlation_pairs if pair["Significant"] == "Yes"
    )
    high_corr_count = sum(
        1 for pair in correlation_pairs if pair["Abs Correlation"] > 0.7
    )
    very_high_corr_count = sum(
        1 for pair in correlation_pairs if pair["Abs Correlation"] > 0.9
    )

    return {
        "Total Feature Pairs": len(all_correlations),
        "Pairs Above Threshold": len(correlation_pairs),
        "Significant Correlations": significant_count,
        "High Correlations (>0.7)": high_corr_count,
        "Very High Correlations (>0.9)": very_high_corr_count,
        "Mean Absolute Correlation": np.mean(all_correlations),
        "Max Absolute Correlation": np.max(all_correlations),
        "Median Absolute Correlation": np.median(all_correlations),
    }


@tags("tabular_data", "statistics", "correlation")
@tasks("classification", "regression", "clustering")
def CorrelationAnalysis(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    significance_level: float = 0.05,
    min_correlation: float = 0.1,
) -> Dict[str, Any]:
    """
    Performs comprehensive correlation analysis with significance testing for numerical features.

    ### Purpose

    This test conducts detailed correlation analysis between numerical features, including
    correlation coefficients, significance testing, and identification of significant
    relationships. It helps identify multicollinearity, feature relationships, and
    potential redundancies in the dataset.

    ### Test Mechanism

    The test computes correlation coefficients using the specified method and performs
    statistical significance testing for each correlation pair. It provides:
    - Correlation matrix with significance indicators
    - List of significant correlations above threshold
    - Summary statistics about correlation patterns
    - Identification of highly correlated feature pairs

    ### Signs of High Risk

    - Very high correlations (>0.9) indicating potential multicollinearity
    - Many significant correlations suggesting complex feature interactions
    - Features with no significant correlations to others (potential isolation)
    - Unexpected correlation patterns contradicting domain knowledge

    ### Strengths

    - Provides statistical significance testing for correlations
    - Supports multiple correlation methods (Pearson, Spearman, Kendall)
    - Identifies potentially problematic high correlations
    - Filters results by minimum correlation threshold
    - Comprehensive summary of correlation patterns

    ### Limitations

    - Limited to numerical features only
    - Cannot detect non-linear relationships (except with Spearman)
    - Significance testing assumes certain distributional properties
    - Correlation does not imply causation
    """
    # Validate and prepare data
    data = _validate_and_prepare_data(dataset, columns)

    # Compute correlation matrices
    corr_matrix, p_matrix = _compute_correlation_matrices(data, method)

    # Create correlation pairs
    correlation_pairs = _create_correlation_pairs(
        corr_matrix, p_matrix, significance_level, min_correlation
    )

    # Build results
    results = {}
    if correlation_pairs:
        results["Correlation Pairs"] = format_records(pd.DataFrame(correlation_pairs))

    # Create summary statistics
    summary_stats = _create_summary_statistics(corr_matrix, correlation_pairs)
    results["Summary Statistics"] = format_records(pd.DataFrame([summary_stats]))

    return results


def _compute_pearson_with_pvalues(data):
    """Compute Pearson correlation with p-values"""
    n_vars = data.shape[1]
    corr_matrix = data.corr(method="pearson")
    p_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j:
                _, p_val = stats.pearsonr(data[col1], data[col2])
                p_matrix.iloc[i, j] = p_val

    return corr_matrix, p_matrix


def _compute_spearman_with_pvalues(data):
    """Compute Spearman correlation with p-values"""
    n_vars = data.shape[1]
    corr_matrix = data.corr(method="spearman")
    p_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j:
                _, p_val = stats.spearmanr(data[col1], data[col2])
                p_matrix.iloc[i, j] = p_val

    return corr_matrix, p_matrix


def _compute_kendall_with_pvalues(data):
    """Compute Kendall correlation with p-values"""
    n_vars = data.shape[1]
    corr_matrix = data.corr(method="kendall")
    p_matrix = pd.DataFrame(
        np.zeros((n_vars, n_vars)), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j:
                _, p_val = stats.kendalltau(data[col1], data[col2])
                p_matrix.iloc[i, j] = p_val

    return corr_matrix, p_matrix


def _correlation_strength(abs_corr):
    """Classify correlation strength"""
    if abs_corr >= 0.9:
        return "Very Strong"
    elif abs_corr >= 0.7:
        return "Strong"
    elif abs_corr >= 0.5:
        return "Moderate"
    elif abs_corr >= 0.3:
        return "Weak"
    else:
        return "Very Weak"
