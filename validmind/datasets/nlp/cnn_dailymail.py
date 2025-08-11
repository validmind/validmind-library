# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os
import textwrap
from typing import Optional, Tuple

import pandas as pd
from IPython.display import HTML, display
from tabulate import tabulate

from validmind.errors import MissingDependencyError

# Define column names
text_column = "article"
target_column = "highlights"
gpt_35_prediction_column = "gpt_35_prediction"
t5_prediction = "t5_prediction"
bert_embedding_prediction_column = "bert_embedding_model_prediction"

# Define the path to the dataset directory
current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")


def load_data(
    source: str = "online", dataset_size: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from either online source or offline files.

    Args:
        source: 'online' for online data, 'offline' for offline data. Defaults to 'online'.
        dataset_size: Applicable if source is 'offline'. '300k' or '500k' for dataset size. Defaults to None.

    Returns:
        Tuple containing (train_df, test_df) DataFrames with the loaded data.
    """
    if source == "online":
        try:
            from datasets import load_dataset
        except ImportError as e:
            if "datasets" in str(e):
                raise MissingDependencyError(
                    "Missing required package `datasets` for CNN Daily Mail. "
                    "Please run `pip install validmind[datasets]` or "
                    "`pip install datasets` to use CNN Daily Mail dataset",
                    required_dependencies=["datasets"],
                    extra="datasets",
                ) from e
            raise e

        # Load online data without predictions
        cnn_dataset = load_dataset("cnn_dailymail", "3.0.0")
        train_df = cnn_dataset["train"].to_pandas()
        test_df = cnn_dataset["test"].to_pandas()

        # Process the DataFrame to include necessary columns
        train_df = train_df[["article", "highlights"]]
        test_df = test_df[["article", "highlights"]]

        return train_df, test_df

    elif source == "offline":
        # Determine the file name based on the dataset size
        if dataset_size == "100":
            data_file_name = "cnn_dailymail_100_with_predictions.csv"
        elif dataset_size == "500":
            data_file_name = "cnn_dailymail_500_with_predictions.csv"
        else:
            raise ValueError("Invalid dataset_size specified. Choose '100' or '500'.")

        # Construct the file path
        data_file = os.path.join(dataset_path, data_file_name)

        # Load the dataset
        df = pd.read_csv(data_file)
        df = df[
            [
                "article",
                "highlights",
                "gpt_35_prediction",
                "t5_prediction",
                "bert_embedding_model_prediction",
            ]
        ]

        train_df = df.sample(frac=0.7, random_state=42)
        test_df = df.drop(train_df.index)
        return train_df, test_df

    else:
        raise ValueError("Invalid source specified. Choose 'online' or 'offline'.")


def _format_cell_text(text, width=50):
    """Private function to format a cell's text."""
    return "\n".join([textwrap.fill(line, width=width) for line in text.split("\n")])


def _format_dataframe_for_tabulate(df):
    """Private function to format the entire DataFrame for tabulation."""
    df_out = df.copy()

    # Format all string columns
    for column in df_out.columns:
        # Check if column is of type object (likely strings)
        if df_out[column].dtype == object:
            df_out[column] = df_out[column].apply(_format_cell_text)
    return df_out


def _dataframe_to_html_table(df):
    """Private function to convert a DataFrame to an HTML table."""
    headers = df.columns.tolist()
    table_data = df.values.tolist()
    return tabulate(table_data, headers=headers, tablefmt="html")


def display_nice(df, num_rows=None):
    """Primary function to format and display a DataFrame."""
    if num_rows is not None:
        df = df.head(num_rows)
    formatted_df = _format_dataframe_for_tabulate(df)
    html_table = _dataframe_to_html_table(formatted_df)
    display(HTML(html_table))
