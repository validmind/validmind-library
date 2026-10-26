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
def ResidualBiasBySegment(
    dataset: VMDataset,
    model: VMModel,
    segment_columns: Optional[List[SegmentColumn]] = None,
    n_bins: int = 5,
    max_categories: int = 20,
    min_segment_size: int = 30,
    confidence_level: float = 0.95,
    adjust_for_multiple_segments: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Finds segments of the data in which a regression model systematically over- or
    under-predicts, by measuring the mean residual in every segment with a confidence
    interval.

    ### Purpose

    A model can be unbiased on average and badly biased for a group: over-predicting one
    segment and under-predicting another cancel out in every portfolio-level metric. Bias
    concentrated in a segment is often the most material model risk — it means the model
    systematically misprices or misreserves that group. Segmenting on columns the model does
    *not* use is especially revealing, because it exposes effects the model cannot see.

    ### Test Mechanism

    Residuals are `actual - predicted` (positive = under-prediction). The data is split into
    segments for every column in `segment_columns` (default: every column of the dataset
    except the target, including extra columns not used by the model):

    - numeric columns with more than `max_categories` distinct values are cut into
    `n_bins` quantile bins; other columns are used as categories;
    - a list of columns, for example `["IsStudent", "Income"]`, segments on their
    combinations, to look for interactions.

    For each segment the test reports its size, mean actual and predicted values, the mean
    residual with its confidence interval, and the bias relative to the mean actual value.
    A segment is **flagged** when its interval excludes zero. With
    `adjust_for_multiple_segments` (default), the confidence level is Bonferroni-adjusted
    for the number of segments tested, so that scanning many segments does not flag some by
    chance. Segments smaller than `min_segment_size` are reported but not assessed.

    The test passes when no segment is flagged.

    ### Signs of High Risk

    - Flagged segments with a material relative bias, especially large or high-exposure ones.
    - Opposite biases in complementary segments (for example under-predicting one group and
    over-predicting the rest), which average to zero overall.
    - Bias along a column the model does not use: an omitted effect.

    ### Strengths

    - Works for any regression model with predictions assigned to the dataset.
    - Scans every column by default, including ones the model does not use.
    - Controls false alarms when many segments are tested.

    ### Limitations

    - Bins and categories are fixed in advance; a biased group that cuts across them can be
    diluted.
    - The Bonferroni adjustment is conservative: with many segments, small real biases may
    not be flagged.
    - Uses a normal approximation for the interval, which needs reasonably sized segments.
    """
    df = dataset.df
    columns = segment_columns_or_default(dataset, segment_columns)
    actual = np.asarray(dataset.y, dtype=float)
    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    residuals = actual - predicted

    segments = list(iter_segments(df, columns, n_bins, max_categories))
    assessed = sum(mask.sum() >= min_segment_size for _, _, mask in segments)
    z = z_value(confidence_level, assessed if adjust_for_multiple_segments else 1)

    rows = []
    for column, segment, mask in segments:
        n = int(mask.sum())
        bias, half_width = mean_and_half_width(residuals[mask], z)
        assessable = n >= min_segment_size
        flagged = bool(assessable and abs(bias) > half_width)
        mean_actual = float(actual[mask].mean())
        rows.append(
            {
                "Segment Column": column,
                "Segment": segment,
                "Count": n,
                "Mean Actual": mean_actual,
                "Mean Predicted": float(predicted[mask].mean()),
                "Mean Residual": bias,
                "CI Lower": bias - half_width,
                "CI Upper": bias + half_width,
                "Relative Bias": bias / mean_actual if mean_actual else np.nan,
                "Result": (
                    "Too small to assess"
                    if not assessable
                    else ("Biased" if flagged else "OK")
                ),
            }
        )
    table = pd.DataFrame(rows)

    labels = table["Segment Column"] + ": " + table["Segment"]
    colors = table["Result"].map(
        {"Biased": "#c62828", "OK": "#3949ab", "Too small to assess": "#bdbdbd"}
    )
    fig = go.Figure(
        go.Scatter(
            x=table["Mean Residual"],
            y=labels,
            mode="markers",
            marker=dict(size=8, color=colors),
            error_x=dict(
                type="data",
                array=table["CI Upper"] - table["Mean Residual"],
                arrayminus=table["Mean Residual"] - table["CI Lower"],
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#757575")
    fig.update_layout(
        title=(
            f"Mean Residual by Segment (positive = under-prediction) — "
            f"{model.input_id} on {dataset.input_id}"
        ),
        xaxis_title="Mean residual (actual − predicted)",
        yaxis=dict(autorange="reversed"),
        height=200 + 22 * len(table),
    )

    passed = bool((table["Result"] != "Biased").all())

    return (
        table.round(4),
        fig,
        passed,
        RawData(
            residuals=residuals,
            z_value=z,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
