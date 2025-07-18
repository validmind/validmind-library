# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("tabular_data", "visualization", "categorical_data")
@tasks("classification")
def TargetRateBarPlots(dataset: VMDataset) -> Tuple[go.Figure, RawData]:
    """
    Generates bar plots visualizing the default rates of categorical features for a classification machine learning
    model.

    ### Purpose

    This test, implemented as a metric, is designed to provide an intuitive, graphical summary of the decision-making
    patterns exhibited by a categorical classification machine learning model. The model's performance is evaluated
    using bar plots depicting the ratio of target rates—meaning the proportion of positive classes—for different
    categorical inputs. This allows for an easy, at-a-glance understanding of the model's accuracy.

    ### Test Mechanism

    The test involves creating a pair of bar plots for each categorical feature in the dataset. The first plot depicts
    the frequency of each category in the dataset, with each category visually distinguished by its unique color. The
    second plot shows the mean target rate of each category (sourced from the "default_column"). Plotly, a Python
    library, is used to generate these plots, with distinct plots created for each feature. If no specific columns are
    selected, the test will generate plots for each categorical column in the dataset.

    ### Signs of High Risk

    - Inconsistent or non-binary values in the "default_column" could complicate or render impossible the calculation
    of average target rates.
    - Particularly low or high target rates for a specific category might suggest that the model is misclassifying
    instances of that category.

    ### Strengths

    - This test offers a visually interpretable breakdown of the model's decisions, providing an easy way to spot
    irregularities, inconsistencies, or patterns.
    - Its flexibility allows for the inspection of one or multiple columns, as needed.

    ### Limitations

    - The readability of the bar plots drops as the number of distinct categories increases in the dataset, which can
    make them harder to understand and less useful.
    """
    if np.unique(dataset.df[dataset.target_column]).size != 2:
        raise SkipTestError(
            f"Target column '{dataset.target_column}' is not binary. "
            "This test only works for binary classification tasks."
        )

    if len(dataset.feature_columns_categorical) == 0:
        raise SkipTestError("No categorical columns found in the dataset")

    df = dataset.df
    figures = []
    raw_data = []

    for col in dataset.feature_columns_categorical:
        # Calculate counts and default rate for each category
        counts = df[col].value_counts()
        default_rate = df.groupby(col)[dataset.target_column].mean()
        raw_data.append({"column": col, "counts": counts, "default_rate": default_rate})

        fig = make_subplots(
            rows=1,
            cols=2,
        )

        # Left plot: Counts
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                name="Counts",
                marker_color="#6699cc",
            ),
            row=1,
            col=1,
        )
        # Right plot: Default rate
        fig.add_trace(
            go.Bar(
                x=default_rate.index,
                y=default_rate.values,
                name="Target Rate",
                marker_color="orange",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text=col,
            autosize=False,
            width=500,
            height=400,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        figures.append(fig)

    return (
        *figures,
        RawData(target_rates_by_category=raw_data, dataset=dataset.input_id),
    )
