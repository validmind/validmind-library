# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

# URLs or file paths for online and offline data
data_file = os.path.join(dataset_path, "lending_club_biased.csv.gz")

target_column = "loan_status"
protected_classes = ["Gender", "Race", "Marital_Status"]

drop_columns = ["total_pymnt", "id", "verification_status", "purpose"]

score_params = {
    "target_score": 600,
    "target_odds": 50,
    "pdo": 20,
}


def load_data():
    """
    Load data from the specified CSV file.

    :return: DataFrame containing the loaded data.
    """

    print(f"Loading data from: {data_file}")
    # Since we know the offline_data_file path ends with '.zip', we replace it with '.csv.gz'
    gzip_file_path = data_file.replace(".zip", ".csv.gz")
    # Read the CSV file directly from the .gz archive
    df = pd.read_csv(gzip_file_path, compression="gzip")
    print("Data loaded successfully.")
    df = _clean_data(df)

    return df


def _clean_data(df):
    df = df.copy()
    print("Loading the raw dataset:")
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    # Drop columns not relevant for this model
    print(f"Dropping columns not relevant for this model: {drop_columns}")
    df = df.drop(columns=drop_columns)
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    # Drop rows with missing target values
    df.dropna(subset=[target_column], inplace=True)
    print("Dropping rows with missing target values:")
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    # Drop columns with more than N percent missing values
    missing_values = df.isnull().mean()
    df = df.loc[:, missing_values < 0.7]
    print("Dropping columns with more than 70% missing values:")
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    # Drop columns with only one unique value
    unique_values = df.nunique()
    df = df.loc[:, unique_values > 1]
    print("Dropping columns with only one unique value:")
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    return df


def preprocess(df):
    df = df.copy()

    # Convert the target variable to integer type for modeling.
    df[target_column] = df[target_column].astype(int)

    # Identify and encode categorical variables for modeling purposes
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        le = LabelEncoder()
        df[f"{column}_encoded"] = le.fit_transform(df[column])
        label_encoders[column] = le
        df = df.drop(columns=[column])  # Remove the original column

    print(f"Encoding categorical variables: {list(categorical_columns)}")
    print(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
    )

    return df


def split(df, test_size=0.3):
    df = df.copy()

    # Splitting the dataset into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Calculate and print details for the training dataset
    print(
        f"Training Dataset:\nRows: {train_df.shape[0]}\nColumns: {train_df.shape[1]}\nMissing values: {train_df.isnull().sum().sum()}\n"
    )

    # Calculate and print details for the test dataset
    print(
        f"Test Dataset:\nRows: {test_df.shape[0]}\nColumns: {test_df.shape[1]}\nMissing values: {test_df.isnull().sum().sum()}\n"
    )

    return train_df, test_df


def compute_scores(probabilities):

    target_score = score_params["target_score"]
    target_odds = score_params["target_odds"]
    pdo = score_params["pdo"]

    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))

    scores = offset + factor * np.log(probabilities / (1 - probabilities))

    return scores
