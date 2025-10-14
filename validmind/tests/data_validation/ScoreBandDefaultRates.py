# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk", "scorecard")
@tasks("classification")
def ScoreBandDefaultRates(
    dataset: VMDataset,
    model: VMModel,
    score_column: str = "score",
    score_bands: list = None,
) -> Tuple[pd.DataFrame, RawData]:
    """
    Analyzes default rates and population distribution across credit score bands.

    ### Purpose

    The Score Band Default Rates test evaluates the discriminatory power of credit scores by analyzing
    default rates across different score bands. This helps validate score effectiveness, supports
    policy decisions, and provides insights into portfolio risk distribution.

    ### Test Mechanism

    The test segments the score distribution into bands and calculates key metrics for each band:
    1. Population count and percentage in each band
    2. Default rate within each band
    3. Cumulative statistics across bands
    The results show how well the scores separate good and bad accounts.

    ### Signs of High Risk

    - Non-monotonic default rates across score bands
    - Insufficient population in critical score bands
    - Unexpected default rates for score ranges
    - High concentration in specific score bands
    - Similar default rates across adjacent bands
    - Unstable default rates in key decision bands
    - Extreme population skewness
    - Poor risk separation between bands

    ### Strengths

    - Clear view of score effectiveness
    - Supports policy threshold decisions
    - Easy to interpret and communicate
    - Directly links to business decisions
    - Shows risk segmentation power
    - Identifies potential score issues
    - Helps validate scoring model
    - Supports portfolio monitoring

    ### Limitations

    - Sensitive to band definition choices
    - May mask within-band variations
    - Requires sufficient data in each band
    - Cannot capture non-linear patterns
    - Point-in-time analysis only
    - No temporal trend information
    - Assumes band boundaries are appropriate
    - May oversimplify risk patterns
    """

    if score_column not in dataset.df.columns:
        raise ValueError(
            f"The required column '{score_column}' is not present in the dataset with input_id {dataset.input_id}"
        )

    df = dataset._df.copy()

    # Default score bands if none provided
    if score_bands is None:
        score_bands = [410, 440, 470]

    # Create band labels
    band_labels = [
        f"{score_bands[i]}-{score_bands[i + 1]}" for i in range(len(score_bands) - 1)
    ]
    band_labels.insert(0, f"<{score_bands[0]}")
    band_labels.append(f">{score_bands[-1]}")

    # Bin the scores with infinite upper bound
    df["score_band"] = pd.cut(
        df[score_column], bins=[-np.inf] + score_bands + [np.inf], labels=band_labels
    )

    # Calculate min and max scores for the total row
    min_score = df[score_column].min()
    max_score = df[score_column].max()

    # Get predicted classes (0/1)
    y_pred = dataset.y_pred(model)

    # Calculate metrics by band using target_column name
    results = []
    for band in band_labels:
        band_mask = df["score_band"] == band
        population = band_mask.sum()
        observed_defaults = df[band_mask][dataset.target_column].sum()
        predicted_defaults = y_pred[
            band_mask
        ].sum()  # Sum of 1s gives number of predicted defaults

        results.append(
            {
                "Score Band": band,
                "Population Count": population,
                "Population (%)": population / len(df) * 100,
                "Predicted Default Rate (%)": (
                    predicted_defaults / population * 100 if population > 0 else 0
                ),
                "Observed Default Rate (%)": (
                    observed_defaults / population * 100 if population > 0 else 0
                ),
            }
        )

    # Add total row
    total_population = len(df)
    total_observed = df[dataset.target_column].sum()
    total_predicted = y_pred.sum()  # Total number of predicted defaults

    results.append(
        {
            "Score Band": f"Total ({min_score:.0f}-{max_score:.0f})",
            "Population Count": total_population,
            "Population (%)": sum(r["Population (%)"] for r in results),
            "Predicted Default Rate (%)": total_predicted / total_population * 100,
            "Observed Default Rate (%)": total_observed / total_population * 100,
        }
    )

    return pd.DataFrame(results), RawData(
        results=results, model=model.input_id, dataset=dataset.input_id
    )
