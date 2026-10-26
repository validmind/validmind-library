# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Shared segmentation helpers for the regression segment tests."""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

SegmentColumn = Union[str, Sequence[str]]

MISSING = "Missing"


def segment_columns_or_default(
    dataset, segment_columns: Optional[List[SegmentColumn]]
) -> List[SegmentColumn]:
    """The requested segment columns, or every column of the dataset except the target."""
    if segment_columns is None:
        return [c for c in dataset.df.columns if c != dataset.target_column]

    available = set(dataset.df.columns)
    for spec in segment_columns:
        for column in [spec] if isinstance(spec, str) else spec:
            if column not in available:
                raise ValueError(
                    f"Segment column {column!r} is not in the dataset. Columns the "
                    "model does not use can be added with "
                    "`dataset.add_extra_column(...)` or `extra_columns` in "
                    "`vm.init_dataset`."
                )
    return list(segment_columns)


def segment_labels(values: pd.Series, n_bins: int, max_categories: int) -> pd.Series:
    """Quantile bins for numeric columns with many values; the values themselves otherwise."""
    labels = values
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > max_categories:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
        # qcut nudges the first edge below the minimum; show the real minimum instead
        lowest = values.min()
        # add digits until every bin has its own label (e.g. YYYYMMDD dates need 8)
        for digits in range(4, 16):
            labels = bins.map(
                lambda b: f"{max(b.left, lowest):,.{digits}g} – {b.right:,.{digits}g}",
                na_action="ignore",
            )
            if labels.nunique() == bins.nunique():
                break
    return labels.astype(object).where(values.notna(), MISSING).astype(str)


def iter_segments(df, segment_columns, n_bins, max_categories):
    """Yield (column label, segment label, boolean mask) for every segment."""
    for spec in segment_columns:
        columns = [spec] if isinstance(spec, str) else list(spec)
        labels = [segment_labels(df[c], n_bins, max_categories) for c in columns]
        combined = labels[0]
        for extra in labels[1:]:
            combined = combined + " × " + extra
        column_label = " × ".join(columns)
        for segment in sorted(combined.unique(), key=_natural_key):
            yield column_label, segment, (combined == segment).to_numpy()


def _natural_key(label):
    """Sort numeric bin labels by their lower edge, other labels alphabetically, missing last."""
    head = label.split(" – ")[0].split(" × ")[0].replace(",", "")
    if head == MISSING:
        return (2, 0.0, label)
    try:
        return (0, float(head), label)
    except ValueError:
        return (1, 0.0, label)


def z_value(confidence_level: float, n_comparisons: int = 1) -> float:
    """Two-sided normal quantile, Bonferroni-adjusted for `n_comparisons`."""
    alpha = (1 - confidence_level) / max(n_comparisons, 1)
    return float(stats.norm.ppf(1 - alpha / 2))


def mean_and_half_width(values: np.ndarray, z: float):
    """Mean of `values` and the half-width of its normal-approximation interval."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)) if n else np.nan, np.nan
    return float(values.mean()), float(z * values.std(ddof=1) / np.sqrt(n))
