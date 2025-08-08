# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validmind.errors import MissingDependencyError

try:
    import scorecardpy as sc
except ImportError as e:
    if "scorecardpy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scorecardpy` for WOEBinPlots. "
            "Please run `pip install validmind[credit_risk]` to use these tests",
            required_dependencies=["scorecardpy"],
            extra="credit_risk",
        ) from e
    raise e
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("tabular_data", "visualization", "categorical_data")
@tasks("classification")
def WOEBinPlots(
    dataset: VMDataset,
    breaks_adj: list = None,
    fig_height: int = 600,
    fig_width: int = 500,
) -> Tuple[go.Figure, RawData]:
    """
    Generates visualizations of Weight of Evidence (WoE) and Information Value (IV) for understanding predictive power
    of categorical variables in a data set.

    ### Purpose

    This test is designed to visualize the Weight of Evidence (WoE) and Information Value (IV) for categorical
    variables in a provided dataset. By showcasing the data distribution across different categories of each feature,
    it aids in understanding each variable's predictive power in the context of a classification-based machine learning
    model. Commonly used in credit scoring models, WoE and IV are robust statistical methods for evaluating a
    variable's predictive power.

    ### Test Mechanism

    The test implementation follows defined steps. Initially, it selects non-numeric columns from the dataset and
    changes them to string type, paving the way for accurate binning. It then performs an automated WoE binning
    operation on these selected features, effectively categorizing the potential values of a variable into distinct
    bins. After the binning process, the function generates two separate visualizations (a scatter chart for WoE values
    and a bar chart for IV) for each variable. These visual presentations are formed according to the spread of each
    metric across various categories of each feature.

    ### Signs of High Risk

    - Errors occurring during the binning process.
    - Challenges in converting non-numeric columns into string data type.
    - Misbalance in the distribution of WoE and IV, with certain bins overtaking others conspicuously. This could
    denote that the model is disproportionately dependent on certain variables or categories for predictions, an
    indication of potential risks to its robustness and generalizability.

    ### Strengths

    - Provides a detailed visual representation of the relationship between feature categories and the target variable.
    This grants an intuitive understanding of each feature's contribution to the model.
    - Allows for easy identification of features with high impact, facilitating feature selection and enhancing
    comprehension of the model's decision logic.
    - WoE conversions are monotonic, upholding the rank ordering of the original data points, which simplifies analysis.

    ### Limitations

    - The method is largely reliant on the binning process, and an inappropriate binning threshold or bin number choice
    might result in a misrepresentation of the variable's distribution.
    - While excellent for categorical data, the encoding of continuous variables into categorical can sometimes lead to
    information loss.
    - Extreme or outlier values can dramatically affect the computation of WoE and IV, skewing results.
    - The method requires a sufficient number of events per bin to generate a reliable information value and weight of
    evidence.
    """
    df = dataset.df

    non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns
    df[non_numeric_cols] = df[non_numeric_cols].astype(str)

    try:
        bins = sc.woebin(df, dataset.target_column, breaks_list=breaks_adj)
    except Exception as e:
        raise SkipTestError(f"Error performing binning: {e}") from e

    woe_iv_df = (
        pd.concat(bins.values(), keys=bins.keys())
        .reset_index()
        .drop(columns=["variable"])
        .rename(columns={"level_0": "variable", "level_1": "bin_number"})
    )

    figures = []

    for variable in woe_iv_df["variable"].unique():
        variable_df = woe_iv_df[woe_iv_df["variable"] == variable]

        fig = make_subplots(rows=1, cols=2)

        fig.add_trace(
            go.Bar(
                x=variable_df["bin"],
                y=variable_df["bin_iv"],
                marker_color=px.colors.qualitative.Plotly[: len(variable_df["bin"])],
                hovertemplate="<b>%{x}</b><br>" + "IV: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            ticktext=variable_df["bin"].tolist(),
            tickvals=np.arange(len(variable_df["bin"])),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=variable_df["bin"],
                y=variable_df["woe"],
                mode="lines+markers",
                marker=dict(symbol="circle", size=6),
                hovertemplate="<b>%{x}</b><br>" + "WoE: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(
            ticktext=variable_df["bin"].tolist(),
            tickvals=np.arange(len(variable_df["bin"])),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=f"IV and WoE for {variable}",
            height=fig_height,
            width=fig_width,
            showlegend=False,
        )

        figures.append(fig)

    return (*figures, RawData(woe_iv_data=woe_iv_df, dataset=dataset.input_id))
