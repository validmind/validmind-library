# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Helper script to generate the bundled California housing CSV file.

This script loads the dataset from sklearn and saves it as a CSV file
in the datasets directory for bundling with the repository.

Run this script to generate the bundled dataset file:
    python validmind/datasets/regression/generate_california_housing_csv.py
"""

import os

from sklearn.datasets import fetch_california_housing

# Get the directory paths
current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")
output_file = os.path.join(dataset_path, "california_housing.csv")

# Define columns (matching california_housing.py)
feature_columns = ["HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
target_column = "MedHouseVal"

if __name__ == "__main__":
    print("Loading California housing dataset from sklearn...")

    # Try to load from cache first
    try:
        california_housing = fetch_california_housing(
            as_frame=True, download_if_missing=False
        )
        print("Loaded from sklearn cache.")
    except OSError:
        # Not in cache, try to download
        print("Dataset not in cache, attempting download...")
        try:
            california_housing = fetch_california_housing(as_frame=True)
            print("Downloaded successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nPlease ensure you have internet connectivity and try again.")
            print("Alternatively, manually download from:")
            print("https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz")
            print("and extract to ~/scikit_learn_data/cal_housing/")
            raise

    # Create DataFrame with selected features and target
    df = california_housing.data[feature_columns].copy()
    df[target_column] = california_housing.target.values

    # Ensure dataset directory exists
    os.makedirs(dataset_path, exist_ok=True)

    # Save to CSV
    print(f"\nSaving dataset to: {output_file}")
    df.to_csv(output_file, index=False)

    print(f"✓ Successfully saved {len(df)} rows and {len(df.columns)} columns")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    print("\nDataset is ready to be bundled with the repository!")
