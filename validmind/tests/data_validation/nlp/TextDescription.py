# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import string
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset

try:
    import nltk
    from nltk.corpus import stopwords
except ImportError as e:
    if "nltk" in str(e).lower():
        raise MissingDependencyError(
            "Missing required package `nltk` for TextDescription. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["nltk"],
            extra="nlp",
        ) from e
    raise e


def create_metrics_df(df, text_column, unwanted_tokens, lang):
    stop_words = set(word.lower() for word in stopwords.words(lang))
    unwanted_tokens = set(token.lower() for token in unwanted_tokens)

    results = []

    for text in df[text_column]:
        # pre-process text
        words = nltk.word_tokenize(text)
        filtered_words = [
            word
            for word in words
            if word.lower() not in stop_words
            and word.lower() not in unwanted_tokens
            and word not in string.punctuation
        ]
        sentences = nltk.sent_tokenize(text)

        # calculate metrics
        total_words = len(filtered_words)
        total_sentences = len(sentences)
        avg_sentence_length = round(
            (
                sum(len(sentence.split()) for sentence in sentences) / total_sentences
                if total_sentences
                else 0
            ),
            1,
        )
        total_paragraphs = len(text.split("\n\n"))
        total_unique_words = len(set(filtered_words))
        total_punctuations = sum(1 for word in words if word in string.punctuation)
        lexical_diversity = round(
            total_unique_words / len(filtered_words) if filtered_words else 0, 1
        )

        results.append(
            [
                total_words,
                total_sentences,
                avg_sentence_length,
                total_paragraphs,
                total_unique_words,
                total_punctuations,
                lexical_diversity,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "Total Words",
            "Total Sentences",
            "Avg Sentence Length",
            "Total Paragraphs",
            "Total Unique Words",
            "Total Punctuations",
            "Lexical Diversity",
        ],
    )


@tags("nlp", "text_data", "visualization")
@tasks("text_classification", "text_summarization")
def TextDescription(
    dataset: VMDataset,
    unwanted_tokens: set = {
        "s",
        "s'",
        "mr",
        "ms",
        "mrs",
        "dr",
        "'s",
        " ",
        "''",
        "dollar",
        "us",
        "``",
    },
    lang: str = "english",
) -> Tuple[go.Figure, RawData]:
    """
    Conducts comprehensive textual analysis on a dataset using NLTK to evaluate various parameters and generate
    visualizations.

    ### Purpose

    The TextDescription test aims to conduct a thorough textual analysis of a dataset using the NLTK (Natural Language
    Toolkit) library. It evaluates various metrics such as total words, total sentences, average sentence length, total
    paragraphs, total unique words, most common words, total punctuations, and lexical diversity. The goal is to
    understand the nature of the text and anticipate challenges machine learning models might face in text processing,
    language understanding, or summarization tasks.

    ### Test Mechanism

    The test works by:

    - Parsing the dataset and tokenizing the text into words, sentences, and paragraphs using NLTK.
    - Removing stopwords and unwanted tokens.
    - Calculating parameters like total words, total sentences, average sentence length, total paragraphs, total unique
    words, total punctuations, and lexical diversity.
    - Generating scatter plots to visualize correlations between various metrics (e.g., Total Words vs Total Sentences).

    ### Signs of High Risk

    - Anomalies or increased complexity in lexical diversity.
    - Longer sentences and paragraphs.
    - High uniqueness of words.
    - Large number of unwanted tokens.
    - Missing or erroneous visualizations.

    ### Strengths

    - Essential for pre-processing text data in machine learning models.
    - Provides a comprehensive breakdown of text data, aiding in understanding its complexity.
    - Generates visualizations to help comprehend text structure and complexity.

    ### Limitations

    - Highly dependent on the NLTK library, limiting the test to supported languages.
    - Limited customization for removing undesirable tokens and stop words.
    - Does not consider semantic or grammatical complexities.
    - Assumes well-structured documents, which may result in inaccuracies with poorly formatted text.
    """

    if dataset.text_column is None:
        raise ValueError("A 'text_column' must be provided to run this test.")

    nltk.download("punkt_tab", quiet=True)

    metrics_df = create_metrics_df(
        dataset.df, dataset.text_column, unwanted_tokens, lang
    )

    combinations_to_plot = [
        ("Total Words", "Total Sentences"),
        ("Total Words", "Total Unique Words"),
        ("Total Sentences", "Avg Sentence Length"),
        ("Total Unique Words", "Lexical Diversity"),
    ]

    figures = []

    # Create hist plots for each column
    for column in metrics_df.columns:
        fig = px.histogram(metrics_df, x=column)
        fig.update_layout(bargap=0.2)
        figures.append(fig)

    for metric1, metric2 in combinations_to_plot:
        figures.append(
            px.scatter(
                metrics_df,
                x=metric1,
                y=metric2,
                title=f"Scatter Plot: {metric1} vs {metric2}",
            )
        )

    return (*figures, RawData(metrics_dataframe=metrics_df, dataset=dataset.input_id))
