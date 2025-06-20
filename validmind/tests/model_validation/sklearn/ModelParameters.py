# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional

import pandas as pd

from validmind import tags, tasks
from validmind.vm_models import VMModel


@tags("model_training", "metadata")
@tasks("classification", "regression")
def ModelParameters(
    model: VMModel, model_params: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts and displays model parameters in a structured format for transparency and reproducibility.

    ### Purpose

    The Model Parameters test is designed to provide transparency into model configuration and ensure
    reproducibility of machine learning models. It accomplishes this by extracting and presenting all
    relevant parameters that define the model's behavior, making it easier to audit, validate, and
    reproduce model training.

    ### Test Mechanism

    The test leverages scikit-learn's API convention of get_params() to extract model parameters. It
    produces a structured DataFrame containing parameter names and their corresponding values. For models
    that follow scikit-learn's API (including XGBoost, RandomForest, and other estimators), all
    parameters are automatically extracted and displayed.

    ### Signs of High Risk

    - Missing crucial parameters that should be explicitly set
    - Extreme parameter values that could indicate overfitting (e.g., unlimited tree depth)
    - Inconsistent parameters across different versions of the same model type
    - Parameter combinations known to cause instability or poor performance
    - Default values used for critical parameters that should be tuned

    ### Strengths

    - Universal compatibility with scikit-learn API-compliant models
    - Ensures transparency in model configuration
    - Facilitates model reproducibility and version control
    - Enables systematic parameter auditing
    - Supports both classification and regression models
    - Helps identify potential configuration issues

    ### Limitations

    - Only works with models implementing scikit-learn's get_params() method
    - Cannot capture dynamic parameters set during model training
    - Does not validate parameter values for model-specific appropriateness
    - Parameter meanings and impacts may vary across different model types
    - Cannot detect indirect parameter interactions or their effects on model performance
    """
    # Check if model implements get_params()
    if not hasattr(model.model, "get_params"):
        return pd.DataFrame()

    # Get all model parameters
    params = model.model.get_params()

    # If model_params is None, use all parameters from get_params()
    if model_params is None:
        model_params = sorted(params.keys())  # Sort for consistent ordering

    # Create DataFrame with parameters and their values
    param_df = pd.DataFrame(
        [
            {"Parameter": param, "Value": str(params.get(param, "Not specified"))}
            for param in model_params
            if params.get(param) is not None
        ]
    )

    return param_df
