# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import re
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("nlp", "text_data", "visualization", "frequency_analysis")
@tasks("text_classification", "text_summarization")
def Mentions(dataset: VMDataset, top_mentions: int = 25) -> Tuple[go.Figure, RawData]:
    """
    Calculates and visualizes frequencies of '@' prefixed mentions in a text-based dataset for NLP model analysis.

    ### Purpose

    The "Mentions" test is designed to gauge the quality of data in a Natural Language Processing (NLP) or text-focused
    Machine Learning model. The primary objective is to identify and calculate the frequency of 'mentions' within a
    chosen text column of a dataset. A 'mention' in this context refers to individual text elements that are prefixed
    by '@'. The output of this test reveals the most frequently mentioned entities or usernames, which can be integral
    for applications such as social media analyses or customer sentiment analyses.

    ### Test Mechanism

    The test first verifies the existence of a text column in the provided dataset. It then employs a regular
    expression pattern to extract mentions from the text. Subsequently, the frequency of each unique mention is
    calculated. The test selects the most frequent mentions based on default or user-defined parameters, the default
    being the top 25, for representation. This process of thresholding forms the core of the test. A treemap plot
    visualizes the test results, where the size of each rectangle corresponds to the frequency of a particular mention.

    ### Signs of High Risk

    - The lack of a valid text column in the dataset, which would result in the failure of the test execution.
    - The absence of any mentions within the text data, indicating that there might not be any text associated with
    '@'. This situation could point toward sparse or poor-quality data, thereby hampering the model's generalization or
    learning capabilities.

    ### Strengths

    - The test is specifically optimized for text-based datasets which gives it distinct power in the context of NLP.
    - It enables quick identification and visually appealing representation of the predominant elements or mentions.
    - It can provide crucial insights about the most frequently mentioned entities or usernames.

    ### Limitations

    - The test only recognizes mentions that are prefixed by '@', hence useful textual aspects not preceded by '@'
    might be ignored.
    - This test isn't suited for datasets devoid of textual data.
    - It does not provide insights on less frequently occurring data or outliers, which means potentially significant
    patterns could be overlooked.
    """
    mention_counts = (
        dataset.df[dataset.text_column]
        .apply(lambda x: " ".join(re.findall(r"(?<=@)\w+", x)))
        .value_counts()
    )

    if mention_counts.empty:
        raise SkipTestError("No mentions found in the dataset")

    start_index = 1 if mention_counts.iloc[0] == "" else 0
    end_index = top_mentions + start_index
    mention_counts = mention_counts[start_index:end_index]

    mention_frequencies_df = pd.DataFrame(
        {
            "Scenario": mention_counts.index.tolist(),
            "Percentage": mention_counts.tolist(),
        }
    )

    return (
        px.treemap(
            mention_frequencies_df,
            path=["Scenario"],
            values="Percentage",
            title="Tree of Mentions",
        ),
        RawData(mention_counts=mention_counts, dataset=dataset.input_id),
    )
