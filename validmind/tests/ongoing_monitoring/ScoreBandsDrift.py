# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk", "scorecard")
@tasks("classification")
def ScoreBandsDrift(
    datasets: List[VMDataset],
    model: VMModel,
    score_column: str = "score",
    score_bands: list = None,
    drift_threshold: float = 20.0,
) -> Tuple[Dict[str, pd.DataFrame], bool, RawData]:
    """
    Analyzes drift in population distribution and default rates across score bands.

    ### Purpose

    The Score Bands Drift test is designed to evaluate changes in score-based risk segmentation
    over time. By comparing population distribution and default rates across score bands between
    reference and monitoring datasets, this test helps identify whether the model's risk
    stratification remains stable in production. This is crucial for understanding if the model's
    scoring behavior maintains its intended risk separation and whether specific score ranges
    have experienced significant shifts.

    ### Test Mechanism

    This test proceeds by segmenting scores into predefined bands and analyzing three key metrics
    across these bands: population distribution, predicted default rates, and observed default
    rates. For each band, it computes these metrics for both reference and monitoring datasets
    and quantifies drift as percentage changes. The test provides both detailed band-by-band
    comparisons and overall stability assessment, with special attention to bands showing
    significant drift.

    ### Signs of High Risk

    - Large shifts in population distribution across bands
    - Significant changes in default rates within bands
    - Inconsistent drift patterns between adjacent bands
    - Divergence between predicted and observed rates
    - Systematic shifts in risk concentration
    - Empty or sparse score bands in monitoring data

    ### Strengths

    - Provides comprehensive view of score-based drift
    - Identifies specific score ranges with instability
    - Enables comparison of multiple risk metrics
    - Includes both distribution and performance drift
    - Supports business-relevant score segmentation
    - Maintains interpretable drift thresholds

    ### Limitations

    - Sensitive to choice of score band boundaries
    - Requires sufficient samples in each band
    - Cannot suggest optimal band adjustments
    - May not capture within-band distribution changes
    - Limited to predefined scoring metrics
    - Complex interpretation with multiple drift signals
    """
    # Validate score column
    if score_column not in datasets[0].df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in reference dataset"
        )
    if score_column not in datasets[1].df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in monitoring dataset"
        )

    # Default score bands if none provided
    if score_bands is None:
        score_bands = [410, 440, 470]

    # Create band labels
    band_labels = [
        f"{score_bands[i]}-{score_bands[i + 1]}" for i in range(len(score_bands) - 1)
    ]
    band_labels.insert(0, f"<{score_bands[0]}")
    band_labels.append(f">{score_bands[-1]}")

    # Process reference and monitoring datasets
    def process_dataset(dataset, model):
        df = dataset.df.copy()
        df["score_band"] = pd.cut(
            df[score_column],
            bins=[-np.inf] + score_bands + [np.inf],
            labels=band_labels,
        )
        y_pred = dataset.y_pred(model)

        results = {}
        total_population = len(df)

        # Store min and max scores
        min_score = df[score_column].min()
        max_score = df[score_column].max()

        for band in band_labels:
            band_mask = df["score_band"] == band
            population = band_mask.sum()

            results[band] = {
                "Population (%)": population / total_population * 100,
                "Predicted Default Rate (%)": (
                    y_pred[band_mask].sum() / population * 100 if population > 0 else 0
                ),
                "Observed Default Rate (%)": (
                    df[band_mask][dataset.target_column].sum() / population * 100
                    if population > 0
                    else 0
                ),
            }

        results["min_score"] = min_score
        results["max_score"] = max_score
        return results

    # Get metrics for both datasets
    ref_results = process_dataset(datasets[0], model)
    mon_results = process_dataset(datasets[1], model)

    # Create the three comparison tables
    tables = {}
    all_passed = True

    metrics = [
        ("Population Distribution (%)", "Population (%)"),
        ("Predicted Default Rates (%)", "Predicted Default Rate (%)"),
        ("Observed Default Rates (%)", "Observed Default Rate (%)"),
    ]

    for table_name, metric in metrics:
        rows = []
        metric_passed = True

        for band in band_labels:
            ref_val = ref_results[band][metric]
            mon_val = mon_results[band][metric]

            # Calculate drift - using absolute difference when reference is 0
            drift = (
                abs(mon_val - ref_val)
                if ref_val == 0
                else ((mon_val - ref_val) / abs(ref_val)) * 100
            )
            passed = abs(drift) < drift_threshold
            metric_passed &= passed

            rows.append(
                {
                    "Score Band": band,
                    "Reference": round(ref_val, 4),
                    "Monitoring": round(mon_val, 4),
                    "Drift (%)": round(drift, 2),
                    "Pass/Fail": "Pass" if passed else "Fail",
                }
            )

        # Add total row for all metrics
        if metric == "Population (%)":
            ref_total = 100.0
            mon_total = 100.0
            drift_total = 0.0
            passed_total = True
        else:
            ref_total = sum(
                ref_results[band][metric] * (ref_results[band]["Population (%)"] / 100)
                for band in band_labels
            )
            mon_total = sum(
                mon_results[band][metric] * (mon_results[band]["Population (%)"] / 100)
                for band in band_labels
            )
            # Apply same drift calculation to totals
            drift_total = (
                abs(mon_total - ref_total)
                if ref_total == 0
                else ((mon_total - ref_total) / abs(ref_total)) * 100
            )
            passed_total = abs(drift_total) < drift_threshold

        # Format total row with score ranges
        total_label = (
            f"Total ({ref_results['min_score']:.0f}-{ref_results['max_score']:.0f})"
        )

        rows.append(
            {
                "Score Band": total_label,
                "Reference": round(ref_total, 4),
                "Monitoring": round(mon_total, 4),
                "Drift (%)": round(drift_total, 2),
                "Pass/Fail": "Pass" if passed_total else "Fail",
            }
        )

        metric_passed &= passed_total
        tables[table_name] = pd.DataFrame(rows)
        all_passed &= metric_passed

    # Collect raw data
    raw_data = RawData(
        ref_results=ref_results,
        mon_results=mon_results,
        model=model.input_id,
        dataset_reference=datasets[0].input_id,
        dataset_monitoring=datasets[1].input_id,
    )

    return tables, all_passed, raw_data
