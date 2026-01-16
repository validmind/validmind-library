# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os
from csv import DictReader
from uuid import uuid4

import pandas as pd
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")
drop_columns = ["RFP_Question_ID", "Last_Accessed_At"]
target_column = "RFP_Answer"
categorical_columns = []


def load_data(full_dataset=False):
    documents = []
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            # use csv dict reader to load the csv file
            with open(os.path.join(dataset_path, file)) as f:
                reader = DictReader(f)
                for row in reader:
                    # add a unique id to the row
                    row["id"] = str(uuid4())
                    documents.append(row)

    df = pd.DataFrame(documents)
    df.drop(drop_columns, axis=1, inplace=True)

    return df


def preprocess(df):
    df = df.copy()
    train_df, test_df = train_test_split(df, test_size=0.20)

    return train_df, test_df
