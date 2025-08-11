# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    import nltk
except ImportError as e:
    if "nltk" in str(e).lower():
        raise MissingDependencyError(
            "Missing required package `nltk` for ContextualRecall. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["nltk"],
            extra="nlp",
        ) from e
    raise e
from validmind.tests.utils import validate_prediction
from validmind.vm_models import VMDataset, VMModel


@tags("nlp", "text_data", "visualization")
@tasks("text_classification", "text_summarization")
def ContextualRecall(
    dataset: VMDataset, model: VMModel
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Evaluates a Natural Language Generation model's ability to generate contextually relevant and factually correct
    text, visualizing the results through histograms and bar charts, alongside compiling a comprehensive table of
    descriptive statistics for contextual recall scores.

    ### Purpose

    The Contextual Recall metric is used to evaluate the ability of a natural language generation (NLG) model to
    generate text that appropriately reflects the given context or prompt. It measures the model's capability to
    remember and reproduce the main context in its resulting output. This metric is critical in natural language
    processing tasks, as the coherency and contextuality of the generated text are essential.

    ### Test Mechanism

    The function starts by extracting the true and predicted values from the provided dataset and model. It then
    tokenizes the reference and candidate texts into discernible words or tokens using NLTK. The token overlap between
    the reference and candidate texts is identified, and the Contextual Recall score is computed by dividing the number
    of overlapping tokens by the total number of tokens in the reference text. Scores are calculated for each test
    dataset instance, resulting in an array of scores. These scores are visualized using a histogram and a bar chart to
    show score variations across different rows. Additionally, a table of descriptive statistics (mean, median,
    standard deviation, minimum, and maximum) is compiled for the contextual recall scores, providing a comprehensive
    summary of the model's performance.

    ### Signs of High Risk

    - Low contextual recall scores could indicate that the model is not effectively reflecting the original context in
    its output, leading to incoherent or contextually misaligned text.
    - A consistent trend of low recall scores could suggest underperformance of the model.

    ### Strengths

    - Provides a quantifiable measure of a model's adherence to the context and factual elements of the generated
    narrative.
    - Visual representations (histograms and bar charts) make it easier to interpret the distribution and trends of
    contextual recall scores.
    - Descriptive statistics offer a concise summary of the model's performance in generating contextually relevant
    texts.

    ### Limitations

    - The focus on word overlap could result in high scores for texts that use many common words, even when these texts
    lack coherence or meaningful context.
    - This metric does not consider the order of words, which could lead to overestimated scores for scrambled outputs.
    - Models that effectively use infrequent words might be undervalued, as these words might not overlap as often.
    """

    # download nltk data
    nltk.download("punkt_tab", quiet=True)

    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    validate_prediction(y_true, y_pred)

    score_list = []
    for y_t, y_p in zip(y_true, y_pred):
        # Tokenize the reference and candidate texts
        reference_tokens = nltk.word_tokenize(y_t.lower())
        candidate_tokens = nltk.word_tokenize(y_p.lower())

        # Calculate overlapping tokens
        overlapping_tokens = set(reference_tokens) & set(candidate_tokens)

        # Compute contextual recall
        score_list.append(len(overlapping_tokens) / len(reference_tokens))

    metrics_df = pd.DataFrame(score_list, columns=["Contextual Recall"])
    figures = []

    # Histogram for Contextual Recall
    hist_fig = go.Figure(data=[go.Histogram(x=metrics_df["Contextual Recall"])])
    hist_fig.update_layout(
        title="Contextual Recall Histogram",
        xaxis_title="Contextual Recall",
        yaxis_title="Count",
    )
    figures.append(hist_fig)

    # Bar Chart for Contextual Recall
    bar_fig = go.Figure(
        data=[go.Bar(x=metrics_df.index, y=metrics_df["Contextual Recall"])]
    )
    bar_fig.update_layout(
        title="Contextual Recall Bar Chart",
        xaxis_title="Row Index",
        yaxis_title="Contextual Recall",
    )
    figures.append(bar_fig)

    # Calculate statistics for Contextual Recall
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
            contextual_recall_scores=metrics_df,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
