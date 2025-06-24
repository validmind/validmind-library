# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


@tags("time_series_data", "forecasting", "visualization")
@tasks("regression")
def RegressionModelForecastPlot(
    model: VMModel,
    dataset: VMDataset,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[go.Figure, RawData]:
    """
    Generates plots to visually compare the forecasted outcomes of a regression model against actual observed values over
    a specified date range.

    ### Purpose

    This metric is useful for time-series models or any model where the outcome changes over time, allowing direct
    comparison of predicted vs actual values. It can help identify overfitting or underfitting situations as well as
    general model performance.

    ### Test Mechanism

    This test generates a plot with the x-axis representing the date ranging from the specified "start_date" to the
    "end_date", while the y-axis shows the value of the outcome variable. Two lines are plotted: one representing the
    forecasted values and the other representing the observed values. The "start_date" and "end_date" can be parameters
    of this test; if these parameters are not provided, they are set to the minimum and maximum date available in the
    dataset.

    ### Signs of High Risk

    - High risk or failure signs could be deduced visually from the plots if the forecasted line significantly deviates
    from the observed line, indicating the model's predicted values are not matching actual outcomes.
    - A model that struggles to handle the edge conditions like maximum and minimum data points could also be
    considered a sign of risk.

    ### Strengths

    - Visualization: The plot provides an intuitive and clear illustration of how well the forecast matches the actual
    values, making it straightforward even for non-technical stakeholders to interpret.
    - Flexibility: It allows comparison for multiple models and for specified time periods.
    - Model Evaluation: It can be useful in identifying overfitting or underfitting situations, as these will manifest
    as discrepancies between the forecasted and observed values.

    ### Limitations

    - Interpretation Bias: Interpretation of the plot is subjective and can lead to different conclusions by different
    evaluators.
    - Lack of Precision: Visual representation might not provide precise values of the deviation.
    - Inapplicability: Limited to cases where the order of data points (time-series) matters, it might not be of much
    use in problems that are not related to time series prediction.
    """
    index = dataset.df.index

    start_date = index.min() if start_date is None else pd.Timestamp(start_date)
    end_date = index.max() if end_date is None else pd.Timestamp(end_date)

    if start_date < index.min() or end_date > index.max():
        raise ValueError(
            "start_date and end_date must be within the range of dates in the data"
        )

    fig, ax = plt.subplots()

    ax.plot(index, dataset.y, label="Observed")
    ax.plot(index, dataset.y_pred(model), label="Forecast", color="grey")

    plt.title("Forecast vs Observed")

    # Set the x-axis limits to zoom in/out
    plt.xlim(start_date, end_date)

    plt.legend()

    plt.close()

    return fig, RawData(
        observed_values=dataset.y.tolist(),
        forecast_values=dataset.y_pred(model).tolist(),
        model=model.input_id,
        dataset=dataset.input_id,
    )
