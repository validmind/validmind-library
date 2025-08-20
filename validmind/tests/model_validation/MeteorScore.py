# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.tests.utils import validate_prediction
from validmind.vm_models import VMDataset, VMModel

try:
    import evaluate
except ImportError as e:
    if "evaluate" in str(e):
        raise MissingDependencyError(
            "Missing required package `evaluate` for MeteorScore. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["evaluate"],
            extra="nlp",
        ) from e
    raise e


@tags("nlp", "text_data", "visualization")
@tasks("text_classification", "text_summarization")
def MeteorScore(
    dataset: VMDataset, model: VMModel
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Assesses the quality of machine-generated translations by comparing them to human-produced references using the
    METEOR score, which evaluates precision, recall, and word order.

    ### Purpose

    The METEOR (Metric for Evaluation of Translation with Explicit ORdering) score is designed to evaluate the quality
    of machine translations by comparing them against reference translations. It emphasizes both the accuracy and
    fluency of translations, incorporating precision, recall, and word order into its assessment.

    ### Test Mechanism

    The function starts by extracting the true and predicted values from the provided dataset and model. The METEOR
    score is computed for each pair of machine-generated translation (prediction) and its corresponding human-produced
    reference. This is done by considering unigram matches between the translations, including matches based on surface
    forms, stemmed forms, and synonyms. The score is a combination of unigram precision and recall, adjusted for word
    order through a fragmentation penalty. Scores are compiled into a dataframe, and histograms and bar charts are
    generated to visualize the distribution of METEOR scores. Additionally, a table of descriptive statistics (mean,
    median, standard deviation, minimum, and maximum) is compiled for the METEOR scores, providing a comprehensive
    summary of the model's performance.

    ### Signs of High Risk

    - Lower METEOR scores can indicate a lack of alignment between the machine-generated translations and their
    human-produced references, highlighting potential deficiencies in both the accuracy and fluency of translations.
    - Significant discrepancies in word order or an excessive fragmentation penalty could signal issues with how the
    translation model processes and reconstructs sentence structures, potentially compromising the natural flow of
    translated text.
    - Persistent underperformance across a variety of text types or linguistic contexts might suggest a broader
    inability of the model to adapt to the nuances of different languages or dialects, pointing towards gaps in its
    training or inherent limitations.

    ### Strengths

    - Incorporates a balanced consideration of precision and recall, weighted towards recall to reflect the importance
    of content coverage in translations.
    - Directly accounts for word order, offering a nuanced evaluation of translation fluency beyond simple lexical
    matching.
    - Adapts to various forms of lexical similarity, including synonyms and stemmed forms, allowing for flexible
    matching.

    ### Limitations

    - While comprehensive, the complexity of METEOR's calculation can make it computationally intensive, especially for
    large datasets.
    - The use of external resources for synonym and stemming matching may introduce variability based on the resources'
    quality and relevance to the specific translation task.
    """

    # Extract true and predicted values
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    validate_prediction(y_true, y_pred)

    meteor = evaluate.load("meteor")

    # Calculate METEOR scores
    score_list = []
    for y_t, y_p in zip(y_true, y_pred):
        # Compute the METEOR score
        score = meteor.compute(predictions=[y_p], references=[y_t])
        score_list.append(score["meteor"])

    # Convert scores to a dataframe
    metrics_df = pd.DataFrame(score_list, columns=["METEOR Score"])

    figures = []

    # Histogram for METEOR Score
    hist_fig = go.Figure(data=[go.Histogram(x=metrics_df["METEOR Score"])])
    hist_fig.update_layout(
        title="METEOR Score Histogram",
        xaxis_title="METEOR Score",
        yaxis_title="Count",
    )
    figures.append(hist_fig)

    # Bar Chart for METEOR Score
    bar_fig = go.Figure(data=[go.Bar(x=metrics_df.index, y=metrics_df["METEOR Score"])])
    bar_fig.update_layout(
        title="METEOR Score Bar Chart",
        xaxis_title="Row Index",
        yaxis_title="METEOR Score",
    )
    figures.append(bar_fig)

    # Calculate statistics for METEOR Score
    stats_df = metrics_df.describe().loc[["mean", "50%", "max", "min", "std"]]
    stats_df = stats_df.rename(
        index={
            "mean": "Mean Score",
            "50%": "Median Score",
            "max": "Max Score",
            "min": "Min Score",
            "std": "Standard Deviation",
        }
    ).T
    stats_df["Count"] = len(metrics_df)

    # Create a DataFrame from all collected statistics
    result_df = pd.DataFrame(stats_df).reset_index().rename(columns={"index": "Metric"})

    return (
        result_df,
        *figures,
        RawData(
            meteor_scores=metrics_df, model=model.input_id, dataset=dataset.input_id
        ),
    )
