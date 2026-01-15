# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os
from urllib.error import HTTPError, URLError

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

feature_columns = ["HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
target_column = "MedHouseVal"

# Path to bundled dataset file
bundled_data_file = os.path.join(dataset_path, "california_housing.csv")


def load_data(full_dataset=False, source: str = "bundled"):
    """
    Load the California housing dataset.

    This function first attempts to load from a bundled dataset file in the repository.
    If the bundled file is not available, it falls back to sklearn's fetch function
    (which uses cache or attempts download).

    Args:
        full_dataset: Not currently used, kept for API compatibility.
        source: 'bundled' to load from repository file (default), 'sklearn' to use sklearn's fetch.

    Returns:
        pd.DataFrame: DataFrame containing the California housing data.

    Raises:
        RuntimeError: If the dataset cannot be loaded from bundled file or downloaded.
        FileNotFoundError: If bundled file is requested but not found.
    """
    if source == "bundled":
        # Try to load from bundled dataset file
        if os.path.exists(bundled_data_file):
            df = pd.read_csv(bundled_data_file)
            # Ensure we have the correct columns
            required_cols = feature_columns + [target_column]
            if all(col in df.columns for col in required_cols):
                return df[required_cols]
            else:
                # If columns don't match, fall back to sklearn
                return _load_from_sklearn()
        else:
            # Bundled file not found, fall back to sklearn
            return _load_from_sklearn()
    elif source == "sklearn":
        return _load_from_sklearn()
    else:
        raise ValueError("Invalid source specified. Choose 'bundled' or 'sklearn'.")


def _load_from_sklearn():
    """
    Load the California housing dataset from sklearn.

    This function first attempts to load from sklearn's cache. If the dataset
    is not cached, it attempts to download it. If download fails (e.g., 403 Forbidden),
    it provides helpful error messages.

    Returns:
        pd.DataFrame: DataFrame containing the California housing data.

    Raises:
        RuntimeError: If the dataset cannot be downloaded due to network issues (e.g., 403 Forbidden).
        URLError: If there's a URL-related error when fetching the dataset.
        OSError: If the dataset is not cached and cannot be downloaded.
    """
    # First, try to load from cache (won't attempt download)
    try:
        california_housing = fetch_california_housing(
            as_frame=True, download_if_missing=False
        )
    except OSError:
        # Dataset not in cache, try to download it
        try:
            california_housing = fetch_california_housing(as_frame=True)
        except HTTPError as e:
            if (
                "403" in str(e)
                or "Forbidden" in str(e)
                or (hasattr(e, "code") and e.code == 403)
            ):
                error_msg = (
                    "HTTP 403 Forbidden error when downloading California housing dataset.\n\n"
                    "This is a known issue with sklearn's dataset repository access restrictions.\n\n"
                    "The dataset should be available as a bundled file in the repository.\n"
                    "If you're seeing this error, please ensure the bundled dataset file exists\n"
                    "at: {}\n\n".format(bundled_data_file)
                    + "Alternatively, you can manually download the dataset from:\n"
                    "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz\n"
                    "and extract it to ~/scikit_learn_data/cal_housing/"
                )
                raise RuntimeError(error_msg) from e
            else:
                error_msg = (
                    f"Failed to download California housing dataset: {str(e)}\n\n"
                    "This error typically occurs when:\n"
                    "  1. There's no internet connection\n"
                    "  2. The sklearn dataset repository is temporarily unavailable\n"
                    "  3. There's a network firewall blocking the download\n\n"
                    "The dataset should be automatically cached after the first successful download.\n"
                    "Please check your network connection and try again."
                )
                raise RuntimeError(error_msg) from e
        except URLError as e:
            error_msg = (
                f"URL error when downloading California housing dataset: {str(e)}\n\n"
                "This typically indicates a network connectivity issue.\n"
                "Please check your internet connection and try again."
            )
            raise URLError(error_msg) from e

    df = california_housing.data[feature_columns]
    df = df.copy()
    df[target_column] = california_housing.target.values

    return df


def preprocess(df):
    df = df.copy()

    train_val_df, test_df = train_test_split(df, test_size=0.20)

    # This guarantees a 60/20/20 split
    train_df, validation_df = train_test_split(train_val_df, test_size=0.25)

    return train_df, validation_df, test_df
