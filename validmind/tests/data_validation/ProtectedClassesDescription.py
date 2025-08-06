# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import List, Tuple, Union

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.logging import get_logger

logger = get_logger(__name__)


@tags("bias_and_fairness", "descriptive_statistics")
@tasks("classification", "regression")
def ProtectedClassesDescription(
    dataset, protected_classes=None
) -> Tuple[pd.DataFrame, Union[go.Figure, List[go.Figure]], RawData]:
    """
    Visualizes the distribution of protected classes in the dataset relative to the target variable
    and provides descriptive statistics.

    ### Purpose

    The ProtectedClassesDescription test aims to identify potential biases or significant differences in the
    distribution of target outcomes across different protected classes. This visualization and statistical summary
    help in understanding the relationship between protected attributes and the target variable, which is crucial
    for assessing fairness in machine learning models.

    ### Test Mechanism

    The function creates interactive stacked bar charts for each specified protected class using Plotly.
    Additionally, it generates a single table of descriptive statistics for all protected classes, including:
    - Protected class and category
    - Count and percentage of each category within the protected class
    - Mean, median, and mode of the target variable for each category
    - Standard deviation of the target variable for each category
    - Minimum and maximum values of the target variable for each category

    ### Signs of High Risk

    - Significant imbalances in the distribution of target outcomes across different categories of a protected class.
    - Large disparities in mean, median, or mode of the target variable across categories.
    - Underrepresentation or overrepresentation of certain groups within protected classes.
    - High standard deviations in certain categories, indicating potential volatility or outliers.

    ### Strengths

    - Provides both visual and statistical representation of potential biases in the dataset.
    - Allows for easy identification of imbalances in target variable distribution across protected classes.
    - Interactive plots enable detailed exploration of the data.
    - Consolidated statistical summary provides quantitative measures to complement visual analysis.
    - Applicable to both classification and regression tasks.

    ### Limitations

    - Does not provide advanced statistical measures of bias or fairness.
    - May become cluttered if there are many categories within a protected class or many unique target values.
    - Interpretation may require domain expertise to understand the implications of observed disparities.
    - Does not account for intersectionality or complex interactions between multiple protected attributes.
    """

    if protected_classes is None:
        logger.warning(
            "No protected classes provided. Please pass the 'protected_classes' parameter to run this test."
        )
        return pd.DataFrame()

    figures = []
    all_stats = []

    df = dataset._df
    target = dataset.target_column

    for protected_class in protected_classes:
        # Create the stacked bar chart
        counts = df.groupby([protected_class, target]).size().unstack(fill_value=0)
        fig = go.Figure()
        for col in counts.columns:
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts[col],
                    name=str(col),
                    text=counts[col],
                    textposition="auto",
                )
            )

        fig.update_layout(
            title=f"Distribution of {protected_class} by {target}",
            xaxis_title=protected_class,
            yaxis_title="Count",
            barmode="stack",
            showlegend=True,
            legend_title=target,
        )

        figures.append(fig)

        # Get unique values in the target column
        target_labels = df[target].unique()

        for category in df[protected_class].unique():
            category_data = df[df[protected_class] == category]
            stats = {
                "Protected Class": protected_class,
                "Category": category,
                "Count": len(category_data),
                "Percentage": len(category_data) / len(df) * 100,
            }

            # Add mean for each target label
            for label in target_labels:
                label_data = category_data[category_data[target] == label]
                stats[f"Rate {target}: {label}"] = (
                    len(label_data) / len(category_data) * 100
                )

            all_stats.append(stats)

    # Create a single DataFrame with all statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.round(2)  # Round to 2 decimal places for readability

    # Sort the DataFrame by Protected Class and Count (descending)
    stats_df = stats_df.sort_values(
        ["Protected Class", "Count"], ascending=[True, False]
    )

    return (stats_df, *figures, RawData(dataset=dataset.input_id))
