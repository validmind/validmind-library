# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Threshold based tests
"""

from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd
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
            "Missing required package `nltk` for StopWords. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["nltk"],
            extra="nlp",
        ) from e
    raise e


@tags("nlp", "text_data", "frequency_analysis", "visualization")
@tasks("text_classification", "text_summarization")
def StopWords(
    dataset: VMDataset, min_percent_threshold: float = 0.5, num_words: int = 25
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Evaluates and visualizes the frequency of English stop words in a text dataset against a defined threshold.

    ### Purpose

    The StopWords threshold test is a tool designed for assessing the quality of text data in an ML model. It focuses
    on the identification and analysis of "stop words" in a given dataset. Stop words are frequent, common, yet
    semantically insignificant words (for example: "the", "and", "is") in a language. This test evaluates the
    proportion of stop words to the total word count in the dataset, in essence, scrutinizing the frequency of stop
    word usage. The core objective is to highlight the prevalent stop words based on their usage frequency, which can
    be instrumental in cleaning the data from noise and improving ML model performance.

    ### Test Mechanism

    The StopWords test initiates on receiving an input of a 'VMDataset' object. Absence of such an object will trigger
    an error. The methodology involves inspection of the text column of the VMDataset to create a 'corpus' (a
    collection of written texts). Leveraging the Natural Language Toolkit's (NLTK) stop word repository, the test
    screens the corpus for any stop words and documents their frequency. It further calculates the percentage usage of
    each stop word compared to the total word count in the corpus. This percentage is evaluated against a predefined
    'min_percent_threshold'. If this threshold is breached, the test returns a failed output. Top prevailing stop words
    along with their usage percentages are returned, facilitated by a bar chart visualization of these stop words and
    their frequency.

    ### Signs of High Risk

    - A percentage of any stop words exceeding the predefined 'min_percent_threshold'.
    - High frequency of stop words in the dataset which may adversely affect the application's analytical performance
    due to noise creation.

    ### Strengths

    - The ability to scrutinize and quantify the usage of stop words.
    - Provides insights into potential noise in the text data due to stop words.
    - Directly aids in enhancing model training efficiency.
    - Includes a bar chart visualization feature to easily interpret and action upon the stop words frequency
    information.

    ### Limitations

    - The test only supports English stop words, making it less effective with datasets of other languages.
    - The 'min_percent_threshold' parameter may require fine-tuning for different datasets, impacting the overall
    effectiveness of the test.
    - Contextual use of the stop words within the dataset is not considered, potentially overlooking their significance
    in certain contexts.
    - The test focuses specifically on the frequency of stop words, not providing direct measures of model performance
    or predictive accuracy.
    """

    text_column = dataset.text_column

    def create_corpus(df, text_column):
        corpus = []
        for x in df[text_column].str.split():
            for i in x:
                corpus.append(i)
        return corpus

    corpus = create_corpus(dataset.df, text_column=text_column)

    nltk.download("stopwords", quiet=True)

    stop = set(stopwords.words("english"))
    stop_word_frequencies = defaultdict(int)
    for word in corpus:
        if word in stop:
            stop_word_frequencies[word] += 1

    # Calculate the total number of words in the corpus
    total_words = len(corpus)

    # Calculate the percentage of each word in the corpus
    word_percentages = {}
    for word, count in stop_word_frequencies.items():
        percentage = (count / total_words) * 100
        word_percentages[word] = percentage

    passed = all(word_percentages.values()) < min_percent_threshold
    results = sorted(word_percentages.items(), key=lambda x: x[1], reverse=True)[
        :num_words
    ]

    if not results:
        return passed

    x, y = zip(*results)

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(
        title=f"Stop Words Frequency in '{text_column}'",
        xaxis_title="Stop Words",
        yaxis_title="Percentage (%)",
        xaxis_tickangle=-45,
    )

    return (
        {
            f"Stop words results for column '{text_column}'": pd.DataFrame(
                results, columns=["Word", "Percentage"]
            )
        },
        fig,
        passed,
        RawData(stop_word_frequencies=stop_word_frequencies, total_words=total_words),
    )
