# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import re
from typing import Tuple

import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("nlp", "text_data", "visualization", "frequency_analysis")
@tasks("text_classification", "text_summarization")
def Hashtags(dataset: VMDataset, top_hashtags: int = 25) -> Tuple[go.Figure, RawData]:
    """
    Assesses hashtag frequency in a text column, highlighting usage trends and potential dataset bias or spam.

    ### Purpose

    The Hashtags test is designed to measure the frequency of hashtags used within a given text column in a dataset. It
    is particularly useful for natural language processing tasks such as text classification and text summarization.
    The goal is to identify common trends and patterns in the use of hashtags, which can serve as critical indicators
    or features within a machine learning model.

    ### Test Mechanism

    The test implements a regular expression (regex) to extract all hashtags from the specified text column. For each
    hashtag found, it makes a tally of its occurrences. It then outputs a list of the top N hashtags (default is 25,
    but customizable), sorted by their counts in descending order. The results are also visualized in a bar plot, with
    frequency counts on the y-axis and the corresponding hashtags on the x-axis.

    ### Signs of High Risk

    - A low diversity in the usage of hashtags, as indicated by a few hashtags being used disproportionately more than
    others.
    - Repeated usage of one or few hashtags can be indicative of spam or a biased dataset.
    - If there are no or extremely few hashtags found in the dataset, it perhaps signifies that the text data does not
    contain structured social media data.

    ### Strengths

    - Provides a concise visual representation of the frequency of hashtags, which can be critical for understanding
    trends about a particular topic in text data.
    - Instrumental in tasks specifically related to social media text analytics, such as opinion analysis and trend
    discovery.
    - Adaptable, allowing the flexibility to determine the number of top hashtags to be analyzed.

    ### Limitations

    - Assumes the presence of hashtags and therefore may not be applicable for text datasets that do not contain
    hashtags (e.g., formal documents, scientific literature).
    - Language-specific limitations of hashtag formulations are not taken into account.
    - Does not account for typographical errors, variations, or synonyms in hashtags.
    - Does not provide context or sentiment associated with the hashtags, so the information provided may have limited
    utility on its own.
    """
    hashtags = (
        dataset.df[dataset.text_column]
        .apply(lambda x: re.findall(r"(?<=#)\w+", str(x)))
        .explode()
    )
    top_hashtag_counts = hashtags.value_counts().head(top_hashtags)

    if top_hashtag_counts.empty:
        raise SkipTestError("No hashtags found in the dataset")

    fig = go.Figure(
        data=[go.Bar(x=top_hashtag_counts.index, y=top_hashtag_counts.values)]
    )
    fig.update_layout(
        title="Top Hashtags",
        xaxis_title="Hashtag",
        yaxis_title="Count",
        xaxis_tickangle=-45,
    )

    return fig, RawData(top_hashtag_counts=top_hashtag_counts, dataset=dataset.input_id)
