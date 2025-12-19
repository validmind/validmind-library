# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


def calculate_psi(score_initial, score_new, num_bins=10, mode="fixed"):
    """
    Taken from:
    https://towardsdatascience.com/checking-model-stability-and-population-shift-with-psi-and-csi-6d12af008783
    """
    eps = 1e-4

    # Sort the data
    score_initial.sort()
    score_new.sort()

    # Prepare the bins
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == "fixed":
        bins = [
            min_val + (max_val - min_val) * (i) / num_bins for i in range(num_bins + 1)
        ]
    elif mode == "quantile":
        bins = pd.qcut(score_initial, q=num_bins, retbins=True)[
            1
        ]  # Create the quantiles based on the initial population
    else:
        raise ValueError(
            f"Mode '{mode}' not recognized. Allowed options are 'fixed' and 'quantile'"
        )
    bins[0] = min_val - eps  # Correct the lower boundary
    bins[-1] = max_val + eps  # Correct the higher boundary

    # Bucketize the initial population and count the sample inside each bucket
    bins_initial = pd.cut(score_initial, bins=bins, labels=range(1, num_bins + 1))
    df_initial = pd.DataFrame({"initial": score_initial, "bin": bins_initial})
    grp_initial = df_initial.groupby("bin", observed=True).count()
    grp_initial["percent_initial"] = grp_initial["initial"] / sum(
        grp_initial["initial"]
    )

    # Bucketize the new population and count the sample inside each bucket
    bins_new = pd.cut(score_new, bins=bins, labels=range(1, num_bins + 1))
    df_new = pd.DataFrame({"new": score_new, "bin": bins_new})
    grp_new = df_new.groupby("bin", observed=True).count()
    grp_new["percent_new"] = grp_new["new"] / sum(grp_new["new"])

    # Compare the bins to calculate PSI
    psi_df = grp_initial.join(grp_new, on="bin", how="inner")

    # Add a small value for when the percent is zero
    psi_df["percent_initial"] = psi_df["percent_initial"].apply(
        lambda x: eps if x == 0 else x
    )
    psi_df["percent_new"] = psi_df["percent_new"].apply(lambda x: eps if x == 0 else x)

    # Calculate the psi
    psi_df["psi"] = (psi_df["percent_initial"] - psi_df["percent_new"]) * np.log(
        psi_df["percent_initial"] / psi_df["percent_new"]
    )

    return psi_df.to_dict(orient="records")


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def PopulationStabilityIndex(
    datasets: List[VMDataset], model: VMModel, num_bins: int = 10, mode: str = "fixed"
) -> Tuple[Dict[str, List[Dict[str, float]]], go.Figure, RawData]:
    """
    Assesses the Population Stability Index (PSI) to quantify the stability of an ML model's predictions across
    different datasets.

    ### Purpose

    The Population Stability Index (PSI) serves as a quantitative assessment for evaluating the stability of a machine
    learning model's output distributions when comparing two different datasets. Typically, these would be a
    development and a validation dataset or two datasets collected at different periods. The PSI provides a measurable
    indication of any significant shift in the model's performance over time or noticeable changes in the
    characteristics of the population the model is making predictions for.

    ### Test Mechanism

    The implementation of the PSI in this script involves calculating the PSI for each feature between the training and
    test datasets. Data from both datasets is sorted and placed into either a predetermined number of bins or
    quantiles. The boundaries for these bins are initially determined based on the distribution of the training data.
    The contents of each bin are calculated and their respective proportions determined. Subsequently, the PSI is
    derived for each bin through a logarithmic transformation of the ratio of the proportions of data for each feature
    in the training and test datasets. The PSI, along with the proportions of data in each bin for both datasets, are
    displayed in a summary table, a grouped bar chart, and a scatter plot.

    ### Signs of High Risk

    - A high PSI value is a clear indicator of high risk. Such a value suggests a significant shift in the model
    predictions or severe changes in the characteristics of the underlying population.
    - This ultimately suggests that the model may not be performing as well as expected and that it may be less
    reliable for making future predictions.

    ### Strengths

    - The PSI provides a quantitative measure of the stability of a model over time or across different samples, making
    it an invaluable tool for evaluating changes in a model's performance.
    - It allows for direct comparisons across different features based on the PSI value.
    - The calculation and interpretation of the PSI are straightforward, facilitating its use in model risk management.
    - The use of visual aids such as tables and charts further simplifies the comprehension and interpretation of the
    PSI.

    ### Limitations

    - The PSI test does not account for the interdependence between features: features that are dependent on one
    another may show similar shifts in their distributions, which in turn may result in similar PSI values.
    - The PSI test does not inherently provide insights into why there are differences in distributions or why the PSI
    values may have changed.
    - The test may not handle features with significant outliers adequately.
    - Additionally, the PSI test is performed on model predictions, not on the underlying data distributions which can
    lead to misinterpretations. Any changes in PSI could be due to shifts in the model (model drift), changes in the
    relationships between features and the target variable (concept drift), or both. However, distinguishing between
    these causes is non-trivial.
    """
    if model.library in ["statsmodels", "pytorch", "catboost"]:
        raise SkipTestError(f"Skiping PSI for {model.library} models")

    psi_results = calculate_psi(
        datasets[0].y_prob(model).copy(),
        datasets[1].y_prob(model).copy(),
        num_bins=num_bins,
        mode=mode,
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(range(len(psi_results))),
                y=[d["percent_initial"] for d in psi_results],
                name="Initial",
                marker=dict(color="#DE257E"),
            ),
            go.Bar(
                x=list(range(len(psi_results))),
                y=[d["percent_new"] for d in psi_results],
                name="New",
                marker=dict(color="#E8B1F8"),
            ),
            go.Scatter(
                x=list(range(len(psi_results))),
                y=[d["psi"] for d in psi_results],
                name="PSI",
                yaxis="y2",
                line=dict(color="#257EDE"),
            ),
        ],
        layout=go.Layout(
            title="Population Stability Index (PSI) Plot",
            xaxis=dict(title="Bin"),
            yaxis=dict(title="Population Ratio"),
            yaxis2=dict(
                title="PSI",
                overlaying="y",
                side="right",
                range=[
                    0,
                    max(d["psi"] for d in psi_results) + 0.005,
                ],  # Adjust as needed
            ),
            barmode="group",
        ),
    )

    # sum up the PSI values to get the total values
    total_psi = {
        key: sum(d.get(key, 0) for d in psi_results)
        for key in psi_results[0].keys()
        if isinstance(psi_results[0][key], (int, float))
    }
    psi_results.append(total_psi)

    table_title = f"Population Stability Index for {datasets[0].input_id} and {datasets[1].input_id} Datasets"

    return (
        {
            table_title: [
                {
                    "Bin": (
                        i if i < (len(psi_results) - 1) else "Total"
                    ),  # The last bin is the "Total" bin
                    "Count Initial": values["initial"],
                    "Percent Initial (%)": values["percent_initial"] * 100,
                    "Count New": values["new"],
                    "Percent New (%)": values["percent_new"] * 100,
                    "PSI": values["psi"],
                }
                for i, values in enumerate(psi_results)
            ],
        },
        fig,
        RawData(
            psi_raw=psi_results,
            model=model.input_id,
            datasets=[datasets[0].input_id, datasets[1].input_id],
        ),
    )
