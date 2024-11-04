# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("nlp", "text_data", "visualization", "frequency_analysis")
@tasks("text_classification", "text_summarization")
def CommonWords(dataset: VMDataset):
    """
    Assesses the most frequent non-stopwords in a text column for identifying prevalent language patterns.

    ### Purpose

    The CommonWords metric is used to identify and visualize the most prevalent words within a specified text column of
    a dataset. This provides insights into the prevalent language patterns and vocabulary, especially useful in Natural
    Language Processing (NLP) tasks such as text classification and text summarization.

    ### Test Mechanism

    The test methodology involves splitting the specified text column's entries into words, collating them into a
    corpus, and then counting the frequency of each word using the Counter. The forty most frequently occurring
    non-stopwords are then visualized in a bar chart, where the x-axis represents the words, and the y-axis indicates
    their frequency of occurrence.

    ### Signs of High Risk

    - A lack of distinct words within the list, or the most common words being stopwords.
    - Frequent occurrence of irrelevant or inappropriate words could point out a poorly curated or noisy dataset.
    - An error returned due to the absence of a valid Dataset object, indicating high risk as the metric cannot be
    effectively implemented without it.

    ### Strengths

    - The metric provides clear insights into the language features – specifically word frequency – of unstructured
    text data.
    - It can reveal prominent vocabulary and language patterns, which prove vital for feature extraction in NLP tasks.
    - The visualization helps in quickly capturing the patterns and understanding the data intuitively.

    ### Limitations

    - The test disregards semantic or context-related information as it solely focuses on word frequency.
    - It intentionally ignores stopwords, which might carry necessary significance in certain scenarios.
    - The applicability is limited to English-language text data as English stopwords are used for filtering, hence
    cannot account for data in other languages.
    - The metric requires a valid Dataset object, indicating a dependency condition that limits its broader
    applicability.
    """
    counter = Counter(
        [word for x in dataset.df[dataset.text_column].str.split() for word in x]
    )
    most = counter.most_common()

    x = []
    y = []

    nltk.download("stopwords")
    stop = set(stopwords.words("english"))

    for word, count in most[:40]:
        if word not in stop:
            x.append(word)
            y.append(count)

    fig = plt.figure()
    plt.bar(x, y, color="#17C37B")
    plt.xticks(rotation=90)

    return fig
