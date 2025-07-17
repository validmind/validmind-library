# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel
from validmind.vm_models.result import RawData


@tags("sklearn", "model_performance", "classification")
@tasks("classification")
def CalibrationCurve(
    model: VMModel, dataset: VMDataset, n_bins: int = 10
) -> Tuple[go.Figure, RawData]:
    """
    Evaluates the calibration of probability estimates by comparing predicted probabilities against observed
    frequencies.

    ### Purpose

    The Calibration Curve test assesses how well a model's predicted probabilities align with actual
    observed frequencies. This is crucial for applications requiring accurate probability estimates,
    such as risk assessment, decision-making systems, and cost-sensitive applications where probability
    calibration directly impacts business decisions.

    ### Test Mechanism

    The test uses sklearn's calibration_curve function to:
    1. Sort predictions into bins based on predicted probabilities
    2. Calculate the mean predicted probability in each bin
    3. Compare against the observed frequency of positive cases
    4. Plot the results against the perfect calibration line (y=x)
    The resulting curve shows how well the predicted probabilities match empirical probabilities.

    ### Signs of High Risk

    - Significant deviation from the perfect calibration line
    - Systematic overconfidence (predictions too close to 0 or 1)
    - Systematic underconfidence (predictions clustered around 0.5)
    - Empty or sparse bins indicating poor probability coverage
    - Sharp discontinuities in the calibration curve
    - Different calibration patterns across different probability ranges
    - Consistent over/under estimation in critical probability regions
    - Large confidence intervals in certain probability ranges

    ### Strengths

    - Visual and intuitive interpretation of probability quality
    - Identifies systematic biases in probability estimates
    - Supports probability threshold selection
    - Helps understand model confidence patterns
    - Applicable across different classification models
    - Enables comparison between different models
    - Guides potential need for recalibration
    - Critical for risk-sensitive applications

    ### Limitations

    - Sensitive to the number of bins chosen
    - Requires sufficient samples in each bin for reliable estimates
    - May mask local calibration issues within bins
    - Does not account for feature-dependent calibration issues
    - Limited to binary classification problems
    - Cannot detect all forms of miscalibration
    - Assumes bin boundaries are appropriate for the problem
    - May be affected by class imbalance
    """
    prob_true, prob_pred = calibration_curve(
        dataset.y, dataset.y_prob(model), n_bins=n_bins
    )

    # Create DataFrame for raw data
    raw_data = RawData(
        mean_predicted_probability=prob_pred,
        observed_frequency=prob_true,
        model=model.input_id,
        dataset=dataset.input_id,
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        )
    )

    # Add calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Model Calibration",
            line=dict(color="blue"),
            marker=dict(size=8),
        )
    )

    # Update layout
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    return fig, raw_data
