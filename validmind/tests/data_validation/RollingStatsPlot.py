# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


def plot_rolling_statistics(df, col, window_size):
    rolling_mean = df[col].rolling(window=window_size).mean()
    rolling_std = df[col].rolling(window=window_size).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(rolling_mean)
    ax1.set_title(
        f"Rolling Mean for {col}",
        fontsize=20,
        weight="bold",
    )
    ax1.set_ylabel("")
    ax1.tick_params(axis="both", labelsize=18)
    ax1.legend()

    ax2.plot(rolling_std, label="Rolling Standard Deviation", color="orange")
    ax2.set_title(
        f"Rolling STD for {col}",
        fontsize=20,
        weight="bold",
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="both", labelsize=18)
    ax2.legend()

    return fig


@tags("time_series_data", "visualization", "stationarity")
@tasks("regression")
def RollingStatsPlot(
    dataset: VMDataset, window_size: int = 12
) -> Tuple[plt.Figure, RawData]:
    """
    Evaluates the stationarity of time series data by plotting its rolling mean and standard deviation over a specified
    window.

    ### Purpose

    The `RollingStatsPlot` metric is employed to gauge the stationarity of time series data in a given dataset. This
    metric specifically evaluates the rolling mean and rolling standard deviation of the dataset over a pre-specified
    window size. The rolling mean provides an understanding of the average trend in the data, while the rolling
    standard deviation gauges the volatility of the data within the window. It is critical in preparing time series
    data for modeling as it reveals key insights into data behavior across time.

    ### Test Mechanism

    This mechanism is comprised of two steps. Initially, the rolling mean and standard deviation for each of the
    dataset's columns are calculated over a window size, which can be user-specified or by default set to 12 data
    points. Then, the calculated rolling mean and standard deviation are visualized via separate plots, illustrating
    the trends and volatility in the dataset. A straightforward check is conducted to ensure the existence of columns
    in the dataset, and to verify that the given dataset has been indexed by its date and time—a necessary prerequisite
    for time series analysis.

    ### Signs of High Risk

    - The presence of non-stationary patterns in either the rolling mean or the rolling standard deviation plots, which
    could indicate trends or seasonality in the data that may affect the performance of time series models.
    - Missing columns in the dataset, which would prevent the execution of this metric correctly.
    - The detection of NaN values in the dataset, which may need to be addressed before the metric can proceed
    successfully.

    ### Strengths

    - Offers visualizations of trending behavior and volatility within the data, facilitating a broader understanding
    of the dataset's inherent characteristics.
    - Checks of the dataset's integrity, such as the existence of all required columns and the availability of a
    datetime index.
    - Adjusts to accommodate various window sizes, thus allowing accurate analysis of data with differing temporal
    granularities.
    - Considers each column of the data individually, thereby accommodating multi-feature datasets.

    ### Limitations

    - For all columns, a fixed-size window is utilized. This may not accurately capture patterns in datasets where
    different features may require different optimal window sizes.
    - Requires the dataset to be indexed by date and time, hence it may not be usable for datasets without a timestamp
    index.
    - Primarily serves for data visualization as it does not facilitate any quantitative measures for stationarity,
    such as through statistical tests. Therefore, the interpretation is subjective and depends heavily on modeler
    discretion.
    """
    if not pd.api.types.is_datetime64_any_dtype(dataset.df.index):
        raise SkipTestError("Index must be a datetime type")

    figures = [
        plot_rolling_statistics(
            df=dataset.df.dropna(),
            col=col,
            window_size=window_size,
        )
        for col in dataset.feature_columns
    ]

    return (
        *figures,
        RawData(
            rolling_means_stds={
                col: {
                    "rolling_mean": dataset.df[col].rolling(window=window_size).mean(),
                    "rolling_std": dataset.df[col].rolling(window=window_size).std(),
                }
                for col in dataset.feature_columns
            },
            dataset=dataset.input_id,
        ),
    )
