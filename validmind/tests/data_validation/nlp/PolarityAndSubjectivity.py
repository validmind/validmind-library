# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    from textblob import TextBlob
except ImportError as e:
    if "textblob" in str(e).lower():
        raise MissingDependencyError(
            "Missing required package `textblob` for PolarityAndSubjectivity. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["textblob"],
            extra="nlp",
        ) from e
    raise e


@tags("nlp", "text_data", "data_validation")
@tasks("nlp")
def PolarityAndSubjectivity(
    dataset, threshold_subjectivity=0.5, threshold_polarity=0
) -> Tuple[go.Figure, Dict[str, pd.DataFrame], RawData]:
    """
    Analyzes the polarity and subjectivity of text data within a given dataset to visualize the sentiment distribution.

    ### Purpose

    The Polarity and Subjectivity test is designed to evaluate the sentiment expressed in textual data. By analyzing
    these aspects, it helps to identify the emotional tone and subjectivity of the dataset, which could be crucial in
    understanding customer feedback, social media sentiments, or other text-related data.

    ### Test Mechanism

    This test uses TextBlob to compute the polarity and subjectivity scores of textual data in a given dataset. The
    mechanism includes:

    - Iterating through each text entry in the specified column of the dataset.
    - Applying the TextBlob library to compute the polarity (ranging from -1 for negative sentiment to +1 for positive
    sentiment) and subjectivity (ranging from 0 for objective to 1 for subjective) for each entry.
    - Creating a scatter plot using Plotly to visualize the relationship between polarity and subjectivity.

    ### Signs of High Risk

    - High concentration of negative polarity values indicating prevalent negative sentiments.
    - High subjectivity scores suggesting the text data is largely opinion-based rather than factual.
    - Disproportionate clusters of extreme scores (e.g., many points near -1 or +1 polarity).

    ### Strengths

    - Quantifies sentiment and subjectivity which can provide actionable insights.
    - Visualizes sentiment distribution, aiding in easy interpretation.
    - Utilizes well-established TextBlob library for sentiment analysis.

    ### Limitations

    - Polarity and subjectivity calculations may oversimplify nuanced text sentiments.
    - Reliance on TextBlob which may not be accurate for all domains or contexts.
    - Visualization could become cluttered with very large datasets, making interpretation difficult.
    """
    # Check text column
    if not dataset.text_column:
        raise ValueError("Please set text_column name in the Validmind Dataset object")

    sentiments = dataset.df[dataset.text_column].apply(lambda x: TextBlob(x).sentiment)
    data = pd.DataFrame(
        {
            "polarity": [s.polarity for s in sentiments],
            "subjectivity": [s.subjectivity for s in sentiments],
        }
    )

    # Create a Plotly scatter plot
    fig = px.scatter(
        data_frame=data,
        x="polarity",
        y="subjectivity",
        title="Polarity vs Subjectivity",
    )
    fig.update_traces(textposition="top center")

    # Add threshold lines with names for legend
    fig.add_hline(
        y=threshold_subjectivity,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        name=f"Subjectivity Threshold ({threshold_subjectivity})",
    )
    fig.add_vline(
        x=threshold_polarity,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        name=f"Polarity Threshold ({threshold_polarity})",
    )

    fig.update_layout(
        xaxis_title="Polarity",
        yaxis_title="Subjectivity",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
    )

    # Create Quadrant Distribution table
    quadrant_df = pd.DataFrame(
        {
            "Quadrant": [
                "Subjective - Positive Sentiment",
                "Subjective - Negative Sentiment",
                "Objective - Positive Sentiment",
                "Objective - Negative Sentiment",
            ],
            "Ratio (%)": [
                (
                    (data["polarity"] >= threshold_polarity)
                    & (data["subjectivity"] >= threshold_subjectivity)
                ).mean()
                * 100,
                (
                    (data["polarity"] < threshold_polarity)
                    & (data["subjectivity"] >= threshold_subjectivity)
                ).mean()
                * 100,
                (
                    (data["polarity"] >= threshold_polarity)
                    & (data["subjectivity"] < threshold_subjectivity)
                ).mean()
                * 100,
                (
                    (data["polarity"] < threshold_polarity)
                    & (data["subjectivity"] < threshold_subjectivity)
                ).mean()
                * 100,
            ],
        }
    )

    # Create Statistics table
    stats_df = pd.DataFrame(
        {
            "Metric": ["Polarity", "Subjectivity"],
            "Range": ["[-1, 1]", "[0, 1]"],
            "Mean": [data["polarity"].mean(), data["subjectivity"].mean()],
            "Median": [data["polarity"].median(), data["subjectivity"].median()],
            "Std": [data["polarity"].std(), data["subjectivity"].std()],
            "Min": [data["polarity"].min(), data["subjectivity"].min()],
            "Max": [data["polarity"].max(), data["subjectivity"].max()],
        }
    )

    statistics_tables = {"Quadrant Distribution": quadrant_df, "Statistics": stats_df}

    return (
        fig,
        statistics_tables,
        RawData(sentiment_data=data, dataset=dataset.input_id),
    )
