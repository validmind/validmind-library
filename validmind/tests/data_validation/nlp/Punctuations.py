# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Metrics functions for any Pandas-compatible datasets
"""

import string
from collections import defaultdict

import matplotlib.pyplot as plt

from validmind import tags, tasks
from validmind.vm_models import VMDataset


def create_corpus(df, text_column):
    corpus = []
    for x in df[text_column].str.split():
        for i in x:
            corpus.append(i)
        return corpus


@tags("nlp", "text_data", "frequency_analysis", "visualization")
@tasks("text_classification", "text_summarization")
def Punctuations(dataset: VMDataset):
    """
    Analyzes and visualizes the frequency distribution of punctuation usage in a given text dataset.

    ### Purpose

    The Punctuations Metric's primary purpose is to analyze the frequency of punctuation usage within a given text
    dataset. This is often used in Natural Language Processing tasks, such as text classification and text
    summarization.

    ### Test Mechanism

    The test begins by verifying that the input "dataset" is of the type VMDataset. Following that, a corpus is created
    from the dataset by splitting its text on spaces. Each unique punctuation character in the text corpus is then
    tallied. The frequency distribution of each punctuation symbol is visualized as a bar graph, with these results
    being stored as Figures and associated with the main Punctuations object.

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
    text_column = dataset.text_column
    corpus = create_corpus(dataset.df, text_column=text_column)

    special = string.punctuation

    dic = defaultdict(int, {key: 0 for key in special})
    for i in corpus:
        if i in special:
            dic[i] += 1

    fig = plt.figure()
    x, y = zip(*dic.items())
    plt.bar(x, y, color="#17C37B")

    return fig
