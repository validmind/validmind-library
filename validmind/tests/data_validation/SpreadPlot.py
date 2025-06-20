# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("time_series_data", "visualization")
@tasks("regression")
def SpreadPlot(dataset: VMDataset) -> Tuple[plt.Figure, RawData]:
    """
    Assesses potential correlations between pairs of time series variables through visualization to enhance
    understanding of their relationships.

    ### Purpose

    The SpreadPlot test aims to graphically illustrate and analyze the relationships between pairs of time series
    variables within a given dataset. This facilitated understanding helps in identifying and assessing potential time
    series correlations, such as cointegration, between the variables.

    ### Test Mechanism

    The SpreadPlot test computes and represents the spread between each pair of time series variables in the dataset.
    Specifically, the difference between two variables is calculated and presented as a line graph. This process is
    iterated for each unique pair of variables in the dataset, allowing for comprehensive visualization of their
    relationships.

    ### Signs of High Risk

    - Large fluctuations in the spread over a given timespan.
    - Unexpected patterns or trends that may signal potential risks in the underlying correlations between the
    variables.
    - Presence of significant missing data or extreme outlier values, which could potentially skew the spread and
    indicate high risk.

    ### Strengths

    - Allows for thorough visual examination and interpretation of the correlations between time-series pairs.
    - Aids in revealing complex relationships like cointegration.
    - Enhances interpretability by visualizing the relationships, thereby helping in spotting outliers and trends.
    - Capable of handling numerous variable pairs from the dataset through a versatile and adaptable process.

    ### Limitations

    - Primarily serves as a visualization tool and does not offer quantitative measurements or statistics to
    objectively determine relationships.
    - Heavily relies on the quality and granularity of the data—missing data or outliers can notably disturb the
    interpretation of relationships.
    - Can become inefficient or difficult to interpret with a high number of variables due to the profuse number of
    plots.
    - Might not completely capture intricate non-linear relationships between the variables.
    """
    # Validate that the index is datetime
    if not isinstance(dataset.df.index, pd.DatetimeIndex):
        raise SkipTestError("Index must be a datetime type for time series analysis")

    df = dataset.df.dropna()

    # Get all unique pairs of feature columns
    feature_pairs = [
        (dataset.feature_columns[i], dataset.feature_columns[j])
        for i in range(len(dataset.feature_columns))
        for j in range(i + 1, len(dataset.feature_columns))
    ]

    figures = []
    spread_data = {}

    for var1, var2 in feature_pairs:
        fig, ax = plt.subplots()
        fig.suptitle(
            f"Spread between {var1} and {var2}",
            fontsize=20,
            weight="bold",
            y=0.95,
        )

        spread_series = df[var1] - df[var2]
        sns.lineplot(
            data=spread_series,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=18)

        figures.append(fig)
        spread_data[f"{var1}_{var2}_spread"] = spread_series.to_frame(
            name=f"spread_{var1}_{var2}"
        )

    return (*figures, RawData(spread_data=spread_data, dataset=dataset.input_id))
