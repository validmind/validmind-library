# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd

from validmind import tags, tasks
from validmind.utils import get_model_info
from validmind.vm_models import VMModel


@tags("model_training", "metadata")
@tasks("regression", "time_series_forecasting")
def ModelMetadata(model: VMModel) -> pd.DataFrame:
    """
    Compare metadata of different models and generate a summary table with the results.

    **Purpose**: The purpose of this function is to compare the metadata of different models, including information about their architecture, framework, framework version, and programming language.

    **Test Mechanism**: The function retrieves the metadata for each model using `get_model_info`, renames columns according to a predefined set of labels, and compiles this information into a summary table.

    **Signs of High Risk**:
    - Inconsistent or missing metadata across models can indicate potential issues in model documentation or management.
    - Significant differences in framework versions or programming languages might pose challenges in model integration and deployment.

    **Strengths**:
    - Provides a clear comparison of essential model metadata.
    - Standardizes metadata labels for easier interpretation and comparison.
    - Helps identify potential compatibility or consistency issues across models.

    **Limitations**:
    - Assumes that the `get_model_info` function returns all necessary metadata fields.
    - Relies on the correctness and completeness of the metadata provided by each model.
    - Does not include detailed parameter information, focusing instead on high-level metadata.
    """
    column_labels = {
        "architecture": "Modeling Technique",
        "framework": "Modeling Framework",
        "framework_version": "Framework Version",
        "language": "Programming Language",
    }

    def extract_and_rename_metadata(model):
        """
        Extracts metadata for a single model and renames columns based on predefined labels.
        """
        model_info = get_model_info(model)
        renamed_info = {
            column_labels.get(k, k): v for k, v in model_info.items() if k != "params"
        }
        return renamed_info

    # Collect metadata for all models
    metadata_list = [extract_and_rename_metadata(model)]

    # Create a DataFrame from the collected metadata
    metadata_df = pd.DataFrame(metadata_list)

    return metadata_df
