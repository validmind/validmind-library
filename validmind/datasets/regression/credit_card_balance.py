# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Synthetic credit card balance dataset for linear regression.

Each row is a cardholder observed in one quarter; the target is their average monthly
revolving balance. The data is synthetic — see `generate_credit_card_balance_csv.py`
for how it was made.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

feature_columns = [
    "Income",
    "DebtToIncome",
    "CreditScore",
    "CreditLimit",
    "RecentInquiries",
    "NumCards",
    "TenureYears",
    "IsStudent",
    "Region",
]
target_column = "Balance"
time_column = "Quarter"
oot_start = "2025Q3"

bundled_data_file = os.path.join(dataset_path, "credit_card_balance.csv")


def load_data():
    """Load the bundled synthetic credit card balance dataset."""
    return pd.read_csv(bundled_data_file)


def preprocess(df, oot_start=oot_start, test_size=0.2, random_state=42):
    """
    Split into train and test sets from the development period, and an out-of-time
    set from `oot_start` onwards.

    Returns:
        tuple: (train_df, test_df, oot_df)
    """
    development = df[df[time_column] < oot_start]
    oot_df = df[df[time_column] >= oot_start].copy()
    train_df, test_df = train_test_split(
        development.copy(), test_size=test_size, random_state=random_state
    )
    return train_df, test_df, oot_df
