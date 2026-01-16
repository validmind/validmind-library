# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from . import (
    simple_preprocess_booleans,
    simple_preprocess_categoricals,
    simple_preprocess_numericals,
)

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

drop_columns = ["ID"]
boolean_columns = ["SEX"]
categorical_columns = ["MARRIAGE"]

target_column = "DEFAULT"
class_labels = {
    "0": "Did not default",
    "1": "Defaulted",
}


def load_data():
    data_file = os.path.join(dataset_path, "taiwan_credit.csv")
    df = pd.read_csv(data_file)
    return df


def preprocess(df):
    df = df.copy()
    df.drop(drop_columns, axis=1, inplace=True)
    df = simple_preprocess_booleans(df, boolean_columns)
    df = simple_preprocess_categoricals(df, categorical_columns)
    numerical_columns = [
        col
        for col in df.columns
        if col not in boolean_columns + categorical_columns + [target_column]
    ]
    df = simple_preprocess_numericals(df, numerical_columns)

    train_val_df, test_df = train_test_split(df, test_size=0.20)

    # This guarantees a 60/20/20 split
    train_df, validation_df = train_test_split(train_val_df, test_size=0.25)

    return train_df, validation_df, test_df
