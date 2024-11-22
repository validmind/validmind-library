# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import validmind as vm

from . import (
    simple_preprocess_booleans,
    simple_preprocess_categoricals,
    simple_preprocess_numericals,
)

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

drop_columns = ["RowNumber", "CustomerId", "Surname"]
boolean_columns = ["Gender"]
categorical_columns = ["Geography"]

target_column = "Exited"
class_labels = {
    "0": "Did not exit",
    "1": "Exited",
}


def load_data(full_dataset=False):
    data_file = os.path.join(dataset_path, "bank_customer_churn.csv")
    df = pd.read_csv(data_file)

    # Drop these unnecessary columns when loading since they won't be
    # helpful for showing data quality issues
    if full_dataset is False:
        df.drop(drop_columns, axis=1, inplace=True)

    return df


def preprocess(df):
    df = df.copy()
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


def get_demo_test_config(test_suite=None):
    """
    Returns input configuration for the default documentation
    template assigned to this demo model

    The default documentation template uses the following inputs:
    - raw_dataset
    - train_dataset
    - test_dataset
    - model

    We assign the following inputs depending on the input config expected
    by each test:

    - When a test expects a "dataset" we use the raw_dataset
    - When a tets expects "datasets" we use the train_dataset and test_dataset
    - When a test expects a "model" we use the model
    - When a test expects "model" and "dataset" we use the model and test_dataset
    - The only exception is ClassifierPerformance since that runs twice: once
        with the train_dataset (in sample) and once with the test_dataset (out of sample)
    """
    default_config = (test_suite or vm.get_test_suite()).get_default_config()

    for _, test_config in default_config.items():
        if "model" in test_config["inputs"]:
            test_config["inputs"]["model"] = "model"
        if "models" in test_config["inputs"]:
            test_config["inputs"]["models"] = ["model"]
        if "dataset" in test_config["inputs"]:
            if "model" in test_config["inputs"] or "models" in test_config["inputs"]:
                test_config["inputs"]["dataset"] = "test_dataset"
            else:
                test_config["inputs"]["dataset"] = "raw_dataset"
        if "datasets" in test_config["inputs"]:
            test_config["inputs"]["datasets"] = [
                "train_dataset",
                "test_dataset",
            ]

    # ClassifierPerformance is a special case since we run an in-sample and out-of-sample
    # test with two different datasets: train_dataset and test_dataset
    default_config[
        "validmind.model_validation.sklearn.ClassifierPerformance:in_sample"
    ] = {
        "inputs": {
            "model": "model",
            "dataset": "train_dataset",
        }
    }
    default_config[
        "validmind.model_validation.sklearn.ClassifierPerformance:out_of_sample"
    ] = {
        "inputs": {
            "model": "model",
            "dataset": "test_dataset",
        }
    }

    return default_config
