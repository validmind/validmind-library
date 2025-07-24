# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Metrics functions for any Pandas-compatible datasets
"""

import string
from collections import defaultdict
from typing import Tuple

import plotly.graph_objects as go

from validmind import RawData, tags, tasks


@tags("nlp", "text_data", "visualization", "frequency_analysis")
@tasks("text_classification", "text_summarization", "nlp")
def Punctuations(dataset, count_mode="token") -> Tuple[go.Figure, RawData]:
    """
    Analyzes and visualizes the frequency distribution of punctuation usage in a given text dataset.

    ### Purpose

    The Punctuations Metric's primary purpose is to analyze the frequency of punctuation usage within a given text
    dataset. This is often used in Natural Language Processing tasks, such as text classification and text
    summarization.

    ### Test Mechanism

    The test begins by verifying that the input "dataset" is of the type VMDataset. The count_mode parameter must be
    either "token" (counts punctuation marks as individual tokens) or "word" (counts punctuation marks within words).
    Following that, a corpus is created from the dataset by splitting its text on spaces. Each unique punctuation
    character in the text corpus is then tallied. The frequency distribution of each punctuation symbol is visualized
    as a bar graph, with these results being stored as Figures and associated with the main Punctuations object.

    ### Signs of High Risk

    - Excessive or unusual frequency of specific punctuation marks, potentially denoting dubious quality, data
    corruption, or skewed data.

    ### Strengths

    - Provides valuable insights into the distribution of punctuation usage in a text dataset.
    - Important in validating the quality, consistency, and nature of the data.
    - Can provide hints about the style or tonality of the text corpus, such as informal and emotional context
    indicated by frequent exclamation marks.

    ### Limitations

    - Focuses solely on punctuation usage, potentially missing other important textual characteristics.
    - General cultural or tonality assumptions based on punctuation distribution can be misguiding, as these vary
    across different languages and contexts.
    - Less effective with languages that use non-standard or different punctuation.
    - Visualization may lack interpretability when there are many unique punctuation marks in the dataset.
    """
    # Check text column
    if not dataset.text_column:
        raise ValueError("Please set text_column name in the Validmind Dataset object")

    if count_mode not in ["token", "word"]:
        raise ValueError("count_mode parameter must be either 'token' or 'word'")

    corpus = _create_corpus(dataset.df, dataset.text_column)
    punctuation_counts = _count_punctuations(corpus, count_mode)
    fig = _create_punctuation_plot(punctuation_counts)

    return fig, RawData(punctuation_counts=punctuation_counts, dataset=dataset.input_id)


def _create_punctuation_plot(punctuation_counts):
    """Create a bar plot visualization of punctuation frequencies."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(punctuation_counts.keys()),
                y=list(punctuation_counts.values()),
                marker_color="#17C37B",
            )
        ]
    )
    fig.update_layout(
        title="Punctuation Distribution",
        xaxis_title="Punctuation Marks",
        yaxis_title="Frequency",
        showlegend=False,
    )
    return fig


def _create_corpus(df, text_column):
    """Create a corpus from the dataset's text column."""
    corpus = []
    for x in df[text_column].str.split():
        for i in x:
            corpus.append(i)
    return corpus


def _count_punctuations(corpus, count_mode="token"):
    """Count punctuation marks in the corpus based on the specified mode."""
    special = string.punctuation
    dic = defaultdict(int, {key: 0 for key in special})

    if count_mode == "token":
        for i in corpus:
            if i in special:
                dic[i] += 1
    else:  # count_mode == "word"
        for word in corpus:
            for char in word:
                if char in special:
                    dic[char] += 1

    return dic
