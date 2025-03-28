# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Entrypoint for classification datasets.
"""
from typing import List

import pandas as pd

__all__ = [
    "customer_churn",
    "taiwan_credit",
]


def simple_preprocess_booleans(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Preprocess boolean columns.

    Args:
        df (pandas.DataFrame): Dataframe to preprocess.
        columns (list): List of columns to preprocess.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()

    for col in columns:
        unique_values = df[col].unique()
        # Create a dictionary mapping the unique values to integers
        mapping = dict(zip(unique_values, range(len(unique_values))))
        # Map the values to integers using the mapping
        df[col] = df[col].map(mapping)

    return df


def simple_preprocess_categoricals(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Preprocess categorical columns.

    Args:
        df (pandas.DataFrame): Dataframe to preprocess.
        columns (list): List of columns to preprocess.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()

    for col in columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df.drop(col, axis=1, inplace=True)

    return df


def simple_preprocess_numericals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Preprocess numerical columns.

    Args:
        df (pandas.DataFrame): Dataframe to preprocess.
        columns (list): List of columns to preprocess.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()

    for col in columns:
        df[col] = df[col].fillna(df[col].mean())

    return df
