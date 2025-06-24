# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns

from validmind import tags, tasks


@tags("tabular_data", "visualization")
@tasks("classification", "regression")
def ScatterPlot(dataset) -> matplotlib.figure.Figure:
    """
    Assesses visual relationships, patterns, and outliers among features in a dataset through scatter plot matrices.

    ### Purpose

    The ScatterPlot test aims to visually analyze a given dataset by constructing a scatter plot matrix of its
    numerical features. The primary goal is to uncover relationships, patterns, and outliers across different features
    to provide both quantitative and qualitative insights into multidimensional relationships within the dataset. This
    visual assessment aids in understanding the efficacy of the chosen features for model training and their
    suitability.

    ### Test Mechanism

    Using the Seaborn library, the ScatterPlot function creates the scatter plot matrix. The process involves
    retrieving all numerical columns from the dataset and generating a scatter matrix for these columns. The resulting
    scatter plot provides visual representations of feature relationships. The function also adjusts axis labels for
    readability and returns the final plot as a Matplotlib Figure object for further analysis and visualization.

    ### Signs of High Risk

    - The emergence of non-linear or random patterns across different feature pairs, suggesting complex relationships
    unsuitable for linear assumptions.
    - Lack of clear patterns or clusters, indicating weak or non-existent correlations among features, which could
    challenge certain model types.
    - Presence of outliers, as visual outliers can adversely influence the model's performance.

    ### Strengths

    - Provides insight into the multidimensional relationships among multiple features.
    - Assists in identifying trends, correlations, and outliers that could affect model performance.
    - Validates assumptions made during model creation, such as linearity.
    - Versatile for application in both regression and classification tasks.
    - Using Seaborn facilitates an intuitive and detailed visual exploration of data.

    ### Limitations

    - Scatter plot matrices may become cluttered and hard to decipher as the number of features increases.
    - Primarily reveals pairwise relationships and may fail to illuminate complex interactions involving three or more
    features.
    - Being a visual tool, precision in quantitative analysis might be compromised.
    - Outliers not clearly visible in plots can be missed, affecting model performance.
    - Assumes that the dataset can fit into the computer's memory, which might not be valid for extremely large
    datasets.
    """
    g = sns.pairplot(data=dataset.df, diag_kind="kde")

    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation=45)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation=45)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment("right")

    # Get the current figure
    fig = plt.gcf()

    plt.close("all")

    return fig
