# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
    current_path, "..", "..", "..", "notebooks", "datasets", "time_series"
)

target_column = ["loan_rate_A"]
feature_columns = ["loan_rate_B", "loan_rate_C", "loan_rate_D"]
frequency = "MS"
split_option = "train_test"


def load_data():
    data_file = os.path.join(dataset_path, "lending_club_loan_rates.csv")
    df = pd.read_csv(data_file, parse_dates=["DATE"], index_col="DATE")
    return df


def preprocess(df, split_option="train_test_val", train_size=0.6, test_size=0.2):
    """
    Split a time series DataFrame into train, validation, and test sets.

    Parameters:
        df (pandas.DataFrame): The time series DataFrame to be split.
        split_option (str): The split option to choose from: 'train_test_val' (default) or 'train_test'.
        train_size (float): The proportion of the dataset to include in the training set. Default is 0.6.
        test_size (float): The proportion of the dataset to include in the test set. Default is 0.2.

    Returns:
        train_df (pandas.DataFrame): The training set.
        validation_df (pandas.DataFrame): The validation set (only returned if split_option is 'train_test_val').
        test_df (pandas.DataFrame): The test set.

    """
    # Sort the DataFrame by the time column (assuming the time column is the index)
    df = df.sort_index()

    if split_option == "train_test_val":
        # Split the DataFrame into train, validation, and test sets
        train_size = int(len(df) * train_size)
        val_size = int(len(df) * test_size)

        train_df = df.iloc[:train_size]
        validation_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]

        return train_df, validation_df, test_df

    elif split_option == "train_test":
        # Split the DataFrame into train and test sets
        train_size = int(len(df) * train_size)

        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        return train_df, test_df

    else:
        raise ValueError(
            "Invalid split_option. Must be 'train_test_val' or 'train_test'."
        )


def transform(df, transform_func="diff"):
    if transform_func == "diff":
        df = df.diff().dropna()
    return df
