# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


def integrate_diff(series_diff, start_value):
    series_diff = np.array(series_diff)
    series_orig = np.cumsum(series_diff)
    series_orig += start_value

    return series_orig


@tags("time_series_data", "forecasting", "visualization")
@tasks("regression")
def RegressionModelForecastPlotLevels(
    model: VMModel,
    dataset: VMDataset,
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the alignment between forecasted and observed values in regression models through visual plots

    ### Purpose

    This test aims to visually assess the performance of a regression model by comparing its forecasted values against
    the actual observed values for both the raw and transformed (integrated) data. This helps determine the accuracy
    of the model and can help identify overfitting or underfitting. The integration is applied to highlight the trend
    rather than the absolute level.

    ### Test Mechanism

    This test generates two plots:

    - Raw data vs forecast
    - Transformed data vs forecast

    The transformed data is created by performing a cumulative sum on the raw data.

    ### Signs of High Risk

    - Significant deviation between forecasted and observed values.
    - Patterns suggesting overfitting or underfitting.
    - Large discrepancies in the plotted forecasts, indicating potential issues with model generalizability and
    precision.

    ### Strengths

    - Provides an intuitive, visual way to assess multiple regression models, aiding in easier interpretation and
    evaluation of forecast accuracy.

    ### Limitations

    - Relies heavily on visual interpretation, which may vary between individuals.
    - Does not provide a numerical metric to quantify forecast accuracy, relying solely on visual assessment.
    """
    index = dataset.df.index

    if not pd.api.types.is_datetime64_any_dtype(index):
        raise ValueError("Test requires a time series dataset")

    fig, axs = plt.subplots(2, 1)

    y_pred = dataset.y_pred(model)

    # raw data vs forecast
    axs[0].plot(index, dataset.y, label="Observed", color="grey")
    axs[0].plot(index, y_pred, label="Forecast")
    axs[0].set_title("Forecast vs Observed")
    axs[0].legend()

    # transformed data
    dataset_y_transformed = integrate_diff(dataset.y, start_value=dataset.y[0])
    y_pred_transformed = integrate_diff(y_pred, start_value=dataset_y_transformed[0])

    axs[1].plot(
        index,
        dataset_y_transformed,
        label="Observed",
        color="grey",
    )
    axs[1].plot(index, y_pred_transformed, label="Forecast")
    axs[1].set_title("Integrated Forecast vs Observed")
    axs[1].legend()

    plt.close()

    return fig, RawData(
        y_transformed=dataset_y_transformed,
        y_pred_transformed=y_pred_transformed,
        model=model.input_id,
        dataset=dataset.input_id,
    )
