# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import io
import sys
from typing import Any, List, Tuple

import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

try:
    import aequitas.plot as ap
    from aequitas.bias import Bias
    from aequitas.group import Group
    from aequitas.plotting import Plot
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `aequitas` for ProtectedClassesDisparity.",
        required_dependencies=["aequitas"],
    ) from e

logger = get_logger(__name__)


@tags("bias_and_fairness")
@tasks("classification", "regression")
def ProtectedClassesDisparity(
    dataset: VMDataset,
    model: VMModel,
    protected_classes=None,
    disparity_tolerance=1.25,
    metrics=["fnr", "fpr", "tpr"],
) -> Tuple[pd.DataFrame, List[bytes], Any, RawData]:
    """
    Investigates disparities in model performance across different protected class segments.

    ### Purpose

    This test aims to identify and quantify potential biases in model outcomes by comparing various performance metrics
    across different segments of protected classes. It helps in assessing whether the model produces discriminatory
    outcomes for certain groups, which is crucial for ensuring fairness in machine learning models.

    ### Test Mechanism

    The test performs the following steps:
    1. Calculates performance metrics (e.g., false negative rate, false positive rate, true positive rate) for each segment
       of the specified protected classes.
    2. Computes disparity ratios by comparing these metrics between different segments and a reference group.
    3. Generates visualizations showing the disparities and their relation to a user-defined disparity tolerance threshold.
    4. Produces a comprehensive table with various disparity metrics for detailed analysis.

    ### Signs of High Risk

    - Disparity ratios exceeding the specified disparity tolerance threshold.
    - Consistent patterns of higher error rates or lower performance for specific protected class segments.
    - Statistically significant differences in performance metrics across segments.

    ### Strengths

    - Provides a comprehensive view of model fairness across multiple protected attributes and metrics.
    - Allows for easy identification of problematic disparities through visual and tabular representations.
    - Customizable disparity tolerance threshold to align with specific use-case requirements.
    - Applicable to various performance metrics, offering a multi-faceted analysis of model fairness.

    ### Limitations

    - Relies on a predefined reference group for each protected class, which may not always be the most appropriate choice.
    - Does not account for intersectionality between different protected attributes.
    - The interpretation of results may require domain expertise to understand the implications of observed disparities.
    """

    if protected_classes is None:
        logger.warning(
            "No protected classes provided. Please pass the 'protected_classes' parameter to run this test."
        )
        return pd.DataFrame()

    if sys.version_info < (3, 9):
        raise RuntimeError("This test requires Python 3.9 or higher.")

    df = dataset._df

    for protected_class in protected_classes:
        # make the dataset compatible for the python package of interest
        df[protected_class] = pd.Categorical(df[protected_class]).astype("object")

    df["score"] = dataset.y_pred(model).astype(int)
    df["label_value"] = df[dataset.target_column].astype(int)

    # let map the attributes for each protected class
    # default use reference that is most observable for dictionary
    attributes_and_reference_groups = {}
    for protected_class in protected_classes:
        attributes_and_reference_groups.update(
            {protected_class: df[protected_class].value_counts().idxmax()}
        )

    attributes_to_audit = list(attributes_and_reference_groups.keys())

    # Initialize Aequitas
    g = Group()
    b = Bias()
    aqp = Plot()

    columns_to_include = (
        protected_classes + [dataset.target_column] + ["score", "label_value"]
    )

    # get_crosstabs returns a dataframe of the group counts and group value bias metrics.
    xtab, _ = g.get_crosstabs(df[columns_to_include], attr_cols=attributes_to_audit)
    bdf = b.get_disparity_predefined_groups(
        xtab,
        original_df=df[columns_to_include],
        ref_groups_dict=attributes_and_reference_groups,
        alpha=0.05,
        mask_significance=True,
    )

    returns = []  # Renamed to 'returns' for clarity
    for protected_class in protected_classes:
        plot = ap.disparity(
            bdf, metrics, protected_class, fairness_threshold=disparity_tolerance
        )

        buf = io.BytesIO()  # create a bytes array to save the image into in memory
        plot.save(
            buf, format="png"
        )  # as long as the above library is installed, this will work
        returns.append(buf.getvalue())

    string = "_disparity"
    metrics_adj = [x + string for x in metrics]

    table = bdf[["attribute_name", "attribute_value"] + b.list_disparities(bdf)]
    returns.append(aqp.plot_disparity_all(bdf, metrics=metrics_adj))

    return (
        table,
        *returns,
        RawData(model=model.input_id, dataset=dataset.input_id, disparity_data=bdf),
    )
