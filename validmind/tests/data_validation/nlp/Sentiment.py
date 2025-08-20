# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:
    if "nltk" in str(e).lower():
        raise MissingDependencyError(
            "Missing required package `nltk` for Sentiment. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["nltk"],
            extra="nlp",
        ) from e
    raise e


@tags("nlp", "text_data", "data_validation")
@tasks("nlp")
def Sentiment(dataset) -> Tuple[plt.Figure, RawData]:
    """
    Analyzes the sentiment of text data within a dataset using the VADER sentiment analysis tool.

    ### Purpose

    The Sentiment test evaluates the overall sentiment of text data within a dataset. By analyzing sentiment scores, it
    aims to ensure that the model is interpreting text data accurately and is not biased towards a particular sentiment.

    ### Test Mechanism

    This test uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) SentimentIntensityAnalyzer. It processes
    each text entry in a specified column of the dataset to calculate the compound sentiment score, which represents
    the overall sentiment polarity. The distribution of these sentiment scores is then visualized using a KDE (Kernel
    Density Estimation) plot, highlighting any skewness or concentration in sentiment.

    ### Signs of High Risk

    - Extreme polarity in sentiment scores, indicating potential bias.
    - Unusual concentration of sentiment scores in a specific range.
    - Significant deviation from expected sentiment distribution for the given text data.

    ### Strengths

    - Provides a clear visual representation of sentiment distribution.
    - Uses a well-established sentiment analysis tool (VADER).
    - Can handle a wide range of text data, making it flexible for various applications.

    ### Limitations

    - May not capture nuanced or context-specific sentiments.
    - Relies heavily on the accuracy of the VADER sentiment analysis tool.
    - Visualization alone may not provide comprehensive insights into underlying causes of sentiment distribution.
    """
    # Check text column
    if not dataset.text_column:
        raise ValueError("Please set text_column name in the Validmind Dataset object")

    nltk.download("vader_lexicon", quiet=True)
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()

    # Function to get VADER sentiment scores
    def get_vader_sentiment(text):
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score["compound"]

    # Apply the function to each row
    vader_sentiment = dataset.df[dataset.text_column].apply(get_vader_sentiment)

    fig = plt.figure()
    ax = sns.kdeplot(
        x=vader_sentiment,
        fill=True,
        common_norm=False,
        palette="crest",
        alpha=0.5,
        linewidth=0,
    )
    ax.set_title(f"Sentiment score of {dataset.text_column} ")
    ax.set_xlabel("Sentiment score")

    plt.close("all")

    return fig, RawData(
        sentiment_scores=vader_sentiment.tolist(), dataset=dataset.input_id
    )
