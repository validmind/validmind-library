# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks


@tags("visualization")
@tasks("monitoring")
def FeatureDrift(
    datasets, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], feature_columns=None
):
    """
    Evaluates changes in feature distribution over time to identify potential model drift.

    ### Purpose

    The Feature Drift test aims to evaluate how much the distribution of features has shifted over time between two
    datasets, typically training and monitoring datasets. It uses the Population Stability Index (PSI) to quantify this
    change, providing insights into the model’s robustness and the necessity for retraining or feature engineering.

    ### Test Mechanism

    This test calculates the PSI by:

    - Bucketing the distributions of each feature in both datasets.
    - Comparing the percentage of observations in each bucket between the two datasets.
    - Aggregating the differences across all buckets for each feature to produce the PSI score for that feature.

    The PSI score is interpreted as:

    - PSI < 0.1: No significant population change.
    - PSI < 0.2: Moderate population change.
    - PSI >= 0.2: Significant population change.

    ### Signs of High Risk

    - PSI >= 0.2 for any feature, indicating a significant distribution shift.
    - Consistently high PSI scores across multiple features.
    - Sudden spikes in PSI in recent monitoring data compared to historical data.

    ### Strengths

    - Provides a quantitative measure of feature distribution changes.
    - Easily interpretable thresholds for decision-making.
    - Helps in early detection of data drift, prompting timely interventions.

    ### Limitations

    - May not capture more intricate changes in data distribution nuances.
    - Assumes that bucket thresholds (quantiles) adequately represent distribution shifts.
    - PSI score interpretation can be overly simplistic for complex datasets.
    """

    # Feature columns for both datasets should be the same if not given
    default_feature_columns = datasets[0].feature_columns
    feature_columns = feature_columns or default_feature_columns

    x_train_df = datasets[0].x_df()
    x_test_df = datasets[1].x_df()

    quantiles_train = x_train_df[feature_columns].quantile(
        bins, method="single", interpolation="nearest"
    )
    PSI_QUANTILES = quantiles_train.to_dict()

    PSI_BUCKET_FRAC, col, n = get_psi_buckets(
        x_test_df, x_train_df, feature_columns, bins, PSI_QUANTILES
    )

    def nest(d: dict) -> dict:
        result = {}
        for key, value in d.items():
            target = result
            for k in key[:-1]:  # traverse all keys but the last
                target = target.setdefault(k, {})
            target[key[-1]] = value
        return result

    PSI_BUCKET_FRAC = nest(PSI_BUCKET_FRAC)

    PSI_SCORES = {}
    for col in feature_columns:
        psi = 0
        for n in bins:
            actual = PSI_BUCKET_FRAC["test"][col][n]
            expected = PSI_BUCKET_FRAC["train"][col][n]
            psi_of_bucket = (actual - expected) * np.log(
                (actual + 1e-6) / (expected + 1e-6)
            )
            psi += psi_of_bucket
        PSI_SCORES[col] = psi

    psi_df = pd.DataFrame(list(PSI_SCORES.items()), columns=["Features", "PSI Score"])

    psi_df.sort_values(by=["PSI Score"], inplace=True, ascending=False)

    psi_table = [
        {"Features": values["Features"], "PSI Score": values["PSI Score"]}
        for i, values in enumerate(psi_df.to_dict(orient="records"))
    ]

    save_fig = plot_hist(PSI_BUCKET_FRAC, bins)

    final_psi = pd.DataFrame(psi_table)

    return (
        final_psi,
        *save_fig,
        RawData(psi_quantiles=PSI_QUANTILES, psi_bucket_frac=PSI_BUCKET_FRAC),
    )


def get_psi_buckets(x_test_df, x_train_df, feature_columns, bins, PSI_QUANTILES):
    DATA = {"test": x_test_df, "train": x_train_df}
    PSI_BUCKET_FRAC = {}
    for table in DATA.keys():
        total_count = DATA[table].shape[0]
        for col in feature_columns:
            count_sum = 0
            for n in bins:
                if n == 0:
                    bucket_count = (DATA[table][col] < PSI_QUANTILES[col][n]).sum()
                elif n < 9:
                    bucket_count = (
                        total_count
                        - count_sum
                        - ((DATA[table][col] >= PSI_QUANTILES[col][n]).sum())
                    )
                elif n == 9:
                    bucket_count = total_count - count_sum
                count_sum += bucket_count
                PSI_BUCKET_FRAC[table, col, n] = bucket_count / total_count
    return PSI_BUCKET_FRAC, col, n


def plot_hist(PSI_BUCKET_FRAC, bins):
    bin_table_psi = pd.DataFrame(PSI_BUCKET_FRAC)
    save_fig = []
    for i in range(len(bin_table_psi)):

        x = pd.DataFrame(
            bin_table_psi.iloc[i]["test"].items(),
            columns=["Bin", "Population % Reference"],
        )
        y = pd.DataFrame(
            bin_table_psi.iloc[i]["train"].items(),
            columns=["Bin", "Population % Monitoring"],
        )
        xy = x.merge(y, on="Bin")
        xy.index = xy["Bin"]
        xy = xy.drop(columns="Bin", axis=1)
        feature_name = bin_table_psi.index[i]

        n = len(bins)
        r = np.arange(n)
        width = 0.25

        fig = plt.figure()

        plt.bar(
            r,
            xy["Population % Reference"],
            color="b",
            width=width,
            edgecolor="black",
            label="Reference {0}".format(feature_name),
        )
        plt.bar(
            r + width,
            xy["Population % Monitoring"],
            color="g",
            width=width,
            edgecolor="black",
            label="Monitoring {0}".format(feature_name),
        )

        plt.xlabel("Bin")
        plt.ylabel("Population %")
        plt.title("Histogram of Population Differences {0}".format(feature_name))
        plt.legend()
        plt.tight_layout()
        plt.close()
        save_fig.append(fig)
    return save_fig
