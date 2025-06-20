# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


def integrate_diff(series_diff, start_value):
    series_diff = np.asarray(series_diff, dtype=np.float64)  # Convert to float64
    series = np.cumsum(series_diff)
    series += start_value

    return series


@tags("senstivity_analysis", "visualization")
@tasks("regression")
def RegressionModelSensitivityPlot(
    dataset: VMDataset,
    model: VMModel,
    shocks: List[float] = [0.1],
    transformation: Union[str, None] = None,
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the sensitivity of a regression model to changes in independent variables by applying shocks and
    visualizing the impact.

    ### Purpose

    The Regression Sensitivity Plot test is designed to perform sensitivity analysis on regression models. This test
    aims to measure the impact of slight changes (shocks) applied to individual variables on the system's outcome while
    keeping all other variables constant. By doing so, it analyzes the effects of each independent variable on the
    dependent variable within the regression model, helping identify significant risk factors that could substantially
    influence the model's output.

    ### Test Mechanism

    This test operates by initially applying shocks of varying magnitudes, defined by specific parameters, to each of
    the model's features, one at a time. With all other variables held constant, a new prediction is made for each
    dataset subjected to shocks. Any changes in the model's predictions are directly attributed to the shocks applied.
    If the transformation parameter is set to "integrate," initial predictions and target values undergo transformation
    via an integration function before being plotted. Finally, a plot demonstrating observed values against predicted
    values for each model is generated, showcasing a distinct line graph illustrating predictions for each shock.

    ### Signs of High Risk

    - Drastic alterations in model predictions due to minor shocks to an individual variable, indicating high
    sensitivity and potential over-dependence on that variable.
    - Unusually high or unpredictable shifts in response to shocks, suggesting potential model instability or
    overfitting.

    ### Strengths

    - Helps identify variables that strongly influence model outcomes, aiding in understanding feature importance.
    - Generates visual plots, making results easily interpretable even to non-technical stakeholders.
    - Useful in identifying overfitting and detecting unstable models that react excessively to minor variable changes.

    ### Limitations

    - Operates on the assumption that all other variables remain unchanged during the application of a shock, which may
    not reflect real-world interdependencies.
    - Best compatible with linear models and may not effectively evaluate the sensitivity of non-linear models.
    - Provides a visual representation without a numerical risk measure, potentially introducing subjectivity in
    interpretation.
    """
    features_df = dataset.x_df()
    target_df = dataset.y_df()

    shocked_dfs = {"Baseline": features_df}
    for shock in shocks:
        for col in dataset.feature_columns:
            temp_df = features_df.copy()
            temp_df[col] = temp_df[col] * (1 + shock)
            shocked_dfs[f"Shock of {shock} to {col}"] = temp_df

    predictions = {
        label: model.predict(shocked_df) for label, shocked_df in shocked_dfs.items()
    }

    if transformation is None:
        transformed_target = target_df.values
        transformed_predictions = predictions

    elif transformation == "integrate":
        transformed_target = integrate_diff(target_df.values, dataset.y[0])
        transformed_predictions = {
            label: integrate_diff(pred, dataset.y[0])
            for label, pred in predictions.items()
        }

    else:
        raise ValueError(f"Invalid transformation: {transformation}")

    fig = plt.figure()

    plt.plot(target_df.index, transformed_target, label="Observed")

    for label, pred in transformed_predictions.items():
        plt.plot(target_df.index, pred, label=label)

    plt.legend()

    plt.close()

    return fig, RawData(
        transformed_target=transformed_target,
        transformed_predictions=transformed_predictions,
        model=model.input_id,
        dataset=dataset.input_id,
    )
