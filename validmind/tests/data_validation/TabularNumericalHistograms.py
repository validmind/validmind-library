# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.graph_objs as go

from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("tabular_data", "visualization")
@tasks("classification", "regression")
def TabularNumericalHistograms(dataset: VMDataset) -> Tuple[go.Figure]:
    """
    Generates histograms for each numerical feature in a dataset to provide visual insights into data distribution and
    detect potential issues.

    ### Purpose

    The purpose of this test is to provide visual analysis of numerical data through the generation of histograms for
    each numerical feature in the dataset. Histograms aid in the exploratory analysis of data, offering insight into
    the distribution of the data, skewness, presence of outliers, and central tendencies. It helps in understanding if
    the inputs to the model are normally distributed, which is a common assumption in many machine learning algorithms.

    ### Test Mechanism

    This test scans the provided dataset and extracts all the numerical columns. For each numerical column, it
    constructs a histogram using plotly, with 50 bins. The deployment of histograms offers a robust visual aid,
    ensuring unruffled identification and understanding of numerical data distribution patterns.

    ### Signs of High Risk

    - A high degree of skewness
    - Unexpected data distributions
    - Existence of extreme outliers in the histograms

    These may indicate issues with the data that the model is receiving. If data for a numerical feature is expected to
    follow a certain distribution (like a normal distribution) but does not, it could lead to sub-par performance by
    the model. As such these instances should be treated as high-risk indicators.

    ### Strengths

    - Provides a simple, easy-to-interpret visualization of how data for each numerical attribute is distributed.
    - Helps detect skewed values and outliers that could potentially harm the AI model's performance.
    - Can be applied to large datasets and multiple numerical variables conveniently.

    ### Limitations

    - Only works with numerical data, thus ignoring non-numerical or categorical data.
    - Does not analyze relationships between different features, only the individual feature distributions.
    - Is a univariate analysis and may miss patterns or anomalies that only appear when considering multiple variables
    together.
    - Does not provide any insight into how these features affect the output of the model; it is purely an input
    analysis tool.
    """
    if len(dataset.feature_columns_numeric) == 0:
        raise ValueError("No numerical columns found in the dataset")

    df = dataset.df
    figures = []

    for col in dataset.feature_columns_numeric:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], nbinsx=50, name=col))
        fig.update_layout(
            title_text=f"{col}",
            xaxis_title_text="",
            yaxis_title_text="",
            bargap=0.2,
            bargroupgap=0.1,
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )
        figures.append(fig)

    return tuple(figures)
