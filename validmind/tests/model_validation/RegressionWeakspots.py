# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from ._segments import (
    SegmentColumn,
    iter_segments,
    mean_and_half_width,
    segment_columns_or_default,
    z_value,
)


@tags("model_diagnosis", "segment_analysis", "visualization")
@tasks("regression")
def RegressionWeakspots(
    dataset: VMDataset,
    model: VMModel,
    segment_columns: Optional[List[SegmentColumn]] = None,
    n_bins: int = 5,
    max_categories: int = 20,
    min_segment_size: int = 30,
    max_error_ratio: float = 1.25,
    confidence_level: float = 0.95,
    adjust_for_multiple_segments: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Finds segments of the data in which a regression model is materially less accurate
    than overall, by comparing each segment's error with the overall error.

    ### Purpose

    Overall accuracy metrics average over the whole dataset, so a model can look accurate
    while being unreliable for a specific group — high-value customers, a region, a product.
    Where the model is weak determines where it can be relied on and where compensating
    controls or a different model are needed. This is the regression counterpart of a
    weak-spots analysis for classifiers.

    ### Test Mechanism

    Segments are formed as in `ResidualBiasBySegment`: every column in `segment_columns`
    (default: every column of the dataset except the target), numeric columns with more
    than `max_categories` values cut into `n_bins` quantile bins, and lists of columns
    segmenting on their combinations.

    For each segment the test reports the mean absolute error (MAE), root mean squared
    error (RMSE), mean residual, and the ratio of the segment's MAE to the overall MAE. A
    segment is **flagged** as a weak spot when both:

    - its MAE is at least `max_error_ratio` times the overall MAE (material), and
    - the lower bound of its MAE confidence interval is above the overall MAE (not chance),
    with the confidence level Bonferroni-adjusted for the number of segments tested when
    `adjust_for_multiple_segments` is on.

    Segments smaller than `min_segment_size` are reported but not assessed. The test passes
    when there is no weak spot.

    ### Signs of High Risk

    - Weak spots in large or high-exposure segments.
    - Weak spots that coincide with bias in the same segment: the model is both wrong on
    average and noisy there.
    - Error that rises steadily across the bins of one column: a missing non-linearity.

    ### Strengths

    - Works for any regression model with predictions assigned to the dataset.
    - Requires both a material and a statistically clear difference before flagging.

    ### Limitations

    - Segments with naturally larger values (for example high balances) have larger
    absolute errors; read the relative bias and the business context before concluding.
    - Bins are fixed in advance and the Bonferroni adjustment is conservative.
    """
    df = dataset.df
    columns = segment_columns_or_default(dataset, segment_columns)
    actual = np.asarray(dataset.y, dtype=float)
    residuals = actual - np.asarray(dataset.y_pred(model), dtype=float)
    abs_errors = np.abs(residuals)
    overall_mae = float(abs_errors.mean())
    overall_rmse = float(np.sqrt((residuals**2).mean()))

    segments = list(iter_segments(df, columns, n_bins, max_categories))
    assessed = sum(mask.sum() >= min_segment_size for _, _, mask in segments)
    z = z_value(confidence_level, assessed if adjust_for_multiple_segments else 1)

    rows = []
    for column, segment, mask in segments:
        n = int(mask.sum())
        mae, half_width = mean_and_half_width(abs_errors[mask], z)
        ratio = mae / overall_mae if overall_mae else np.nan
        assessable = n >= min_segment_size
        weak = bool(
            assessable and ratio >= max_error_ratio and mae - half_width > overall_mae
        )
        rows.append(
            {
                "Segment Column": column,
                "Segment": segment,
                "Count": n,
                "MAE": mae,
                "MAE CI Lower": mae - half_width,
                "MAE CI Upper": mae + half_width,
                "RMSE": float(np.sqrt((residuals[mask] ** 2).mean())),
                "Mean Residual": float(residuals[mask].mean()),
                "MAE Ratio to Overall": ratio,
                "Result": (
                    "Too small to assess"
                    if not assessable
                    else ("Weak spot" if weak else "OK")
                ),
            }
        )
    table = pd.DataFrame(rows)

    labels = table["Segment Column"] + ": " + table["Segment"]
    colors = table["Result"].map(
        {"Weak spot": "#c62828", "OK": "#3949ab", "Too small to assess": "#bdbdbd"}
    )
    fig = go.Figure(
        go.Bar(
            x=table["MAE Ratio to Overall"],
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=1, line_dash="dash", line_color="#757575")
    fig.add_vline(
        x=max_error_ratio,
        line_dash="dot",
        line_color="#c62828",
        annotation_text=f"{max_error_ratio}× overall MAE",
    )
    fig.update_layout(
        title=(
            f"Segment MAE Relative to Overall MAE ({overall_mae:,.4g}) — "
            f"{model.input_id} on {dataset.input_id}"
        ),
        xaxis_title="Segment MAE / overall MAE",
        yaxis=dict(autorange="reversed"),
        height=200 + 22 * len(table),
    )

    passed = bool((table["Result"] != "Weak spot").all())

    return (
        table.round(4),
        fig,
        passed,
        RawData(
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
