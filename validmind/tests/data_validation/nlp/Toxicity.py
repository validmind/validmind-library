# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    import evaluate
except ImportError as e:
    if "evaluate" in str(e):
        raise MissingDependencyError(
            "Missing required package `evaluate` for Toxicity. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["evaluate"],
            extra="nlp",
        ) from e
    raise e


@tags("nlp", "text_data", "data_validation")
@tasks("nlp")
def Toxicity(dataset) -> Tuple[plt.Figure, RawData]:
    """
    Assesses the toxicity of text data within a dataset to visualize the distribution of toxicity scores.

    ### Purpose

    The Toxicity test aims to evaluate the level of toxic content present in a text dataset by leveraging a pre-trained
    toxicity model. It helps in identifying potentially harmful or offensive language that may negatively impact users
    or stakeholders.

    ### Test Mechanism

    This test uses a pre-trained toxicity evaluation model and applies it to each text entry in the specified column of
    a dataset’s dataframe. The procedure involves:

    - Loading a pre-trained toxicity model.
    - Extracting the text from the specified column in the dataset.
    - Computing toxicity scores for each text entry.
    - Generating a KDE (Kernel Density Estimate) plot to visualize the distribution of these toxicity scores.

    ### Signs of High Risk

    - High concentration of high toxicity scores in the KDE plot.
    - A significant proportion of text entries with toxicity scores above a predefined threshold.
    - Wide distribution of toxicity scores, indicating inconsistency in content quality.

    ### Strengths

    - Provides a visual representation of toxicity distribution, making it easier to identify outliers.
    - Uses a robust pre-trained model for toxicity evaluation.
    - Can process large text datasets efficiently.

    ### Limitations

    - Depends on the accuracy and bias of the pre-trained toxicity model.
    - Does not provide context-specific insights, which may be necessary for nuanced understanding.
    - May not capture all forms of subtle or indirect toxic language.
    """

    # Check text column
    if not dataset.text_column:
        raise ValueError("Please set text_column name in the Validmind Dataset object")

    text_inputs = dataset.df[dataset.text_column].tolist()

    toxicity = evaluate.load("toxicity")
    toxicity_scores = toxicity.compute(predictions=text_inputs)["toxicity"]

    fig = plt.figure()
    ax = sns.kdeplot(
        x=toxicity_scores,
        fill=True,
        common_norm=False,
        palette="crest",
        alpha=0.5,
        linewidth=0,
    )
    ax.set_title(f"Toxicity score of {dataset.text_column}")
    ax.set_xlabel("Toxicity score")

    plt.close()

    return fig, RawData(toxicity_scores=toxicity_scores, dataset=dataset.input_id)
