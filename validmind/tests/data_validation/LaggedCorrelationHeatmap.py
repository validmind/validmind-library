# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset

# Define the 'coolwarm' color scale manually
COOLWARM = [[0, "rgb(95,5,255)"], [0.5, "rgb(255,255,255)"], [1, "rgb(255,5,0)"]]


@tags("time_series_data", "visualization")
@tasks("regression")
def LaggedCorrelationHeatmap(
    dataset: VMDataset, num_lags: int = 10
) -> Tuple[go.Figure, RawData]:
    """
    Assesses and visualizes correlation between target variable and lagged independent variables in a time-series
    dataset.

    ### Purpose

    The LaggedCorrelationHeatmap metric is utilized to appraise and illustrate the correlation between the target
    variable and delayed copies (lags) of independent variables in a time-series dataset. It assists in revealing
    relationships in time-series data where the influence of an independent variable on the dependent variable is not
    immediate but occurs after a period (lags).

    ### Test Mechanism

    To execute this test, Python's Pandas library pairs with Plotly to perform computations and present the
    visualization in the form of a heatmap. The test begins by extracting the target variable and corresponding
    independent variables from the dataset. Then, generation of lags of independent variables takes place, followed by
    the calculation of correlation between these lagged variables and the target variable. The outcome is a correlation
    matrix that gets recorded and illustrated as a heatmap, where different color intensities represent the strength of
    the correlation, making patterns easier to identify.

    ### Signs of High Risk

    - Insignificant correlations across the heatmap, indicating a lack of noteworthy relationships between variables.
    - Correlations that break intuition or previous understanding, suggesting potential issues with the dataset or the
    model.

    ### Strengths

    - This metric serves as an exceptional tool for exploring and visualizing time-dependent relationships between
    features and the target variable in a time-series dataset.
    - It aids in identifying delayed effects that might go unnoticed with other correlation measures.
    - The heatmap offers an intuitive visual representation of time-dependent correlations and influences.

    ### Limitations

    - The metric presumes linear relationships between variables, potentially ignoring non-linear relationships.
    - The correlation considered is linear; therefore, intricate non-linear interactions might be overlooked.
    - The metric is only applicable for time-series data, limiting its utility outside of this context.
    - The number of lags chosen can significantly influence the results; too many lags can render the heatmap difficult
    to interpret, while too few might overlook delayed effects.
    - This metric does not take into account any causal relationships, but merely demonstrates correlation.
    """
    correlations = np.zeros((len(dataset.feature_columns), num_lags + 1))

    for i, ind_var_col in enumerate(dataset.feature_columns):
        for lag in range(num_lags + 1):
            temp_df = pd.DataFrame(
                {
                    dataset.target_column: dataset.df[dataset.target_column],
                    f"{ind_var_col}_lag{lag}": dataset.df[ind_var_col].shift(lag),
                }
            ).dropna()

            corr = temp_df[dataset.target_column].corr(
                temp_df[f"{ind_var_col}_lag{lag}"]
            )

            correlations[i, lag] = corr

    correlation_df = pd.DataFrame(
        correlations,
        columns=[f"{i}" for i in range(num_lags + 1)],
        index=dataset.feature_columns,
    )

    fig = ff.create_annotated_heatmap(
        z=correlation_df.values,
        x=list(correlation_df.columns),
        y=list(correlation_df.index),
        colorscale=COOLWARM,
        annotation_text=correlation_df.round(2).values,
        showscale=True,
    )
    fig.update_layout(
        title={
            "text": f"Correlations between {dataset.target_column} and Lags of Features",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(size=14),
        xaxis_title="Lags",
    )

    return fig, RawData(correlation_matrix=correlation_df, dataset=dataset.input_id)
