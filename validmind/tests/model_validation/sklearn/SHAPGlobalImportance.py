# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import warnings
from typing import Dict, List, Optional, Union
from warnings import filters as _warnings_filters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, UnsupportedModelForSHAPError
from validmind.logging import get_logger
from validmind.models import CatBoostModel, SKlearnModel, StatsModelsModel
from validmind.vm_models import VMDataset, VMModel

try:
    import shap
except ImportError as e:
    if "shap" in str(e):
        raise MissingDependencyError(
            "Missing required package `shap` for SHAPGlobalImportance. "
            "Please run `pip install validmind[explainability]` to use SHAP tests",
            required_dependencies=["shap"],
            extra="explainability",
        ) from e
    raise e

logger = get_logger(__name__)


def select_shap_values(
    shap_values: Union[np.ndarray, List[np.ndarray]],
    class_of_interest: Optional[int] = None,
) -> np.ndarray:
    """Selects SHAP values for binary or multiclass classification.

    For regression models, returns the SHAP values directly as there are no classes.

    Args:
        shap_values: The SHAP values returned by the SHAP explainer. For multiclass
            classification, this will be a list where each element corresponds to a class.
            For regression, this will be a single array of SHAP values.
        class_of_interest: The class index for which to retrieve SHAP values. If None
            (default), the function will assume binary classification and use class 1
            by default.

    Returns:
        The SHAP values for the specified class (classification) or for the regression
        output.

    Raises:
        ValueError: If class_of_interest is specified and is out of bounds for the
            number of classes.
    """
    if not isinstance(shap_values, list):
        # For regression, return the SHAP values as they are
        selected_values = shap_values
    else:
        num_classes = len(shap_values)
        # Default to class 1 for binary classification where no class is specified
        if num_classes == 2 and class_of_interest is None:
            selected_values = shap_values[1]
        # Otherwise, use the specified class_of_interest
        elif class_of_interest is not None and 0 <= class_of_interest < num_classes:
            selected_values = shap_values[class_of_interest]
        else:
            raise ValueError(
                f"Invalid class_of_interest: {class_of_interest}. Must be between 0 and {num_classes - 1}."
            )

    # Add type conversion here to ensure proper float array
    if hasattr(selected_values, "dtype"):
        selected_values = np.array(selected_values, dtype=np.float64)

    return selected_values


def generate_shap_plot(
    type_: str, shap_values: np.ndarray, x_test: Union[np.ndarray, pd.DataFrame]
) -> plt.Figure:
    """Plots two types of SHAP global importance (SHAP).

    Args:
        type_: The type of SHAP plot to generate. Must be "mean" or "summary".
        shap_values: The SHAP values to plot.
        x_test: The test data used to generate the SHAP values.

    Returns:
        The generated plot.
    """
    ax = plt.axes()
    ax.set_facecolor("white")

    if type_ == "mean":
        # Calculate the mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        # Find the maximum mean absolute SHAP value
        max_shap_value = np.max(mean_abs_shap)
        # Normalize all SHAP values based on the top feature
        shap_values = shap_values / max_shap_value * 100

        shap.summary_plot(shap_values, x_test, show=False, plot_type="bar")

        # Customize the plot using matplotlib
        plt.xlabel("Normalized SHAP Value (Percentage)", fontsize=13)
        plt.ylabel("Features", fontsize=13)
        plt.title("Normalized Feature Importance", fontsize=13)
    else:
        shap.summary_plot(shap_values, x_test, show=False)

    fig = plt.gcf()

    plt.close()

    return fig


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "feature_importance",
    "visualization",
)
@tasks("classification", "text_classification")
def SHAPGlobalImportance(
    model: VMModel,
    dataset: VMDataset,
    kernel_explainer_samples: int = 10,
    tree_or_linear_explainer_samples: int = 200,
    class_of_interest: Optional[int] = None,
) -> Dict[str, Union[plt.Figure, Dict[str, float]]]:
    """
    Evaluates and visualizes global feature importance using SHAP values for model explanation and risk identification.

    ### Purpose

    The SHAP (SHapley Additive exPlanations) Global Importance metric aims to elucidate model outcomes by attributing
    them to the contributing features. It assigns a quantifiable global importance to each feature via their respective
    absolute Shapley values, thereby making it suitable for tasks like classification (both binary and multiclass).
    This metric forms an essential part of model risk management.

    ### Test Mechanism

    The exam begins with the selection of a suitable explainer which aligns with the model's type. For tree-based
    models like XGBClassifier, RandomForestClassifier, CatBoostClassifier, TreeExplainer is used whereas for linear
    models like LogisticRegression, XGBRegressor, LinearRegression, it is the LinearExplainer. Once the explainer
    calculates the Shapley values, these values are visualized using two specific graphical representations:

    1. Mean Importance Plot: This graph portrays the significance of individual features based on their absolute
    Shapley values. It calculates the average of these absolute Shapley values across all instances to highlight the
    global importance of features.

    2. Summary Plot: This visual tool combines the feature importance with their effects. Every dot on this chart
    represents a Shapley value for a certain feature in a specific case. The vertical axis is denoted by the feature
    whereas the horizontal one corresponds to the Shapley value. A color gradient indicates the value of the feature,
    gradually changing from low to high. Features are systematically organized in accordance with their importance.

    ### Signs of High Risk

    - Overemphasis on certain features in SHAP importance plots, thus hinting at the possibility of model overfitting
    - Anomalies such as unexpected or illogical features showing high importance, which might suggest that the model's
    decisions are rooted in incorrect or undesirable reasoning
    - A SHAP summary plot filled with high variability or scattered data points, indicating a cause for concern

    ### Strengths

    - SHAP does more than just illustrating global feature significance, it offers a detailed perspective on how
    different features shape the model's decision-making logic for each instance.
    - It provides clear insights into model behavior.

    ### Limitations

    - High-dimensional data can convolute interpretations.
    - Associating importance with tangible real-world impact still involves a certain degree of subjectivity.
    """
    if not isinstance(model, SKlearnModel) or isinstance(
        model, (CatBoostModel, StatsModelsModel)
    ):
        raise UnsupportedModelForSHAPError(
            f"Model {model.class_} is not supported for SHAP importance."
        )

    model_class = model.class_

    # the shap library generates a bunch of annoying warnings that we don't care about
    warnings.filterwarnings("ignore", category=UserWarning)

    if (
        model_class == "XGBClassifier"
        or model_class == "RandomForestClassifier"
        or model_class == "CatBoostClassifier"
        or model_class == "DecisionTreeClassifier"
        or model_class == "RandomForestRegressor"
        or model_class == "GradientBoostingRegressor"
    ):
        explainer = shap.TreeExplainer(model.model)
    elif (
        model_class == "LogisticRegression"
        or model_class == "XGBRegressor"
        or model_class == "LinearRegression"
        or model_class == "LinearSVC"
    ):
        explainer = shap.LinearExplainer(model.model, dataset.x)
    elif model_class == "SVC":
        # KernelExplainer is slow so we use shap.sample to speed it up
        explainer = shap.KernelExplainer(
            model.model.predict,
            shap.sample(
                dataset.x,
                kernel_explainer_samples,
            ),
        )
    else:
        model_class = "<ExternalModel>" if model_class is None else model_class
        raise UnsupportedModelForSHAPError(
            f"Model {model_class} not supported for SHAP importance."
        )

    # KernelExplainer is slow so we use shap.sample to speed it up
    if isinstance(explainer, shap.KernelExplainer):
        shap_sample = shap.sample(
            dataset.x,
            kernel_explainer_samples,
        )
    else:
        shap_sample = dataset.x_df().sample(
            min(
                tree_or_linear_explainer_samples,
                dataset.x_df().shape[0],
            )
        )

    shap_values = explainer.shap_values(shap_sample)
    shap_values = select_shap_values(shap_values, class_of_interest)

    # restore warnings
    _warnings_filters.pop(0)

    return (
        generate_shap_plot("mean", shap_values, shap_sample),
        generate_shap_plot("summary", shap_values, shap_sample),
        RawData(
            shap_values=shap_values,
            shap_sample=shap_sample,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
