# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

feature_columns = [
    "pixel_0_0",
    "pixel_0_1",
    "pixel_0_2",
    "pixel_0_3",
    "pixel_0_4",
    "pixel_0_5",
    "pixel_0_6",
    "pixel_0_7",
    "pixel_1_0",
    "pixel_1_1",
    "pixel_1_2",
    "pixel_1_3",
    "pixel_1_4",
    "pixel_1_5",
    "pixel_1_6",
    "pixel_1_7",
    "pixel_2_0",
    "pixel_2_1",
    "pixel_2_2",
    "pixel_2_3",
    "pixel_2_4",
    "pixel_2_5",
    "pixel_2_6",
    "pixel_2_7",
    "pixel_3_0",
    "pixel_3_1",
    "pixel_3_2",
    "pixel_3_3",
    "pixel_3_4",
    "pixel_3_5",
    "pixel_3_6",
    "pixel_3_7",
    "pixel_4_0",
    "pixel_4_1",
    "pixel_4_2",
    "pixel_4_3",
    "pixel_4_4",
    "pixel_4_5",
    "pixel_4_6",
    "pixel_4_7",
    "pixel_5_0",
    "pixel_5_1",
    "pixel_5_2",
    "pixel_5_3",
    "pixel_5_4",
    "pixel_5_5",
    "pixel_5_6",
    "pixel_5_7",
    "pixel_6_0",
    "pixel_6_1",
    "pixel_6_2",
    "pixel_6_3",
    "pixel_6_4",
    "pixel_6_5",
    "pixel_6_6",
    "pixel_6_7",
    "pixel_7_0",
    "pixel_7_1",
    "pixel_7_2",
    "pixel_7_3",
    "pixel_7_4",
    "pixel_7_5",
    "pixel_7_6",
    "pixel_7_7",
]
target_column = "target"
cluster_labels = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}


def load_data(full_dataset=False):
    digits_data = load_digits(as_frame=True)
    df = digits_data.frame
    return df


def preprocess(df):
    df = df.copy()

    train_val_df, test_df = train_test_split(df, test_size=0.20)

    # This guarantees a 60/20/20 split
    train_df, validation_df = train_test_split(train_val_df, test_size=0.25)

    return train_df, validation_df, test_df
