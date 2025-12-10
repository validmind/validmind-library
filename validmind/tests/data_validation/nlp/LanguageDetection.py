# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    from langdetect import LangDetectException, detect
except ImportError as e:
    if "langdetect" in str(e):
        raise MissingDependencyError(
            "Missing required package `langdetect` for LanguageDetection. "
            "Please run `pip install validmind[nlp]` to use NLP tests",
            required_dependencies=["langdetect"],
            extra="nlp",
        ) from e

    raise e


@tags("nlp", "text_data", "visualization")
@tasks("text_classification", "text_summarization")
def LanguageDetection(dataset) -> Tuple[go.Figure, RawData]:
    """
    Assesses the diversity of languages in a textual dataset by detecting and visualizing the distribution of languages.

    ### Purpose

    The Language Detection test aims to identify and visualize the distribution of languages present within a textual
    dataset. This test helps in understanding the diversity of languages in the data, which is crucial for developing
    and validating multilingual models.

    ### Test Mechanism

    This test operates by:

    - Checking if the dataset has a specified text column.
    - Using a language detection library to determine the language of each text entry in the dataset.
    - Generating a histogram plot of the language distribution, with language codes on the x-axis and their frequencies
    on the y-axis.

    If the text column is not specified, a ValueError is raised to ensure proper dataset configuration.

    ### Signs of High Risk

    - A high proportion of entries returning "Unknown" language codes.
    - Detection of unexpectedly diverse or incorrect language codes, indicating potential data quality issues.
    - Significant imbalance in language distribution, which might indicate potential biases in the dataset.

    ### Strengths

    - Provides a visual representation of language diversity within the dataset.
    - Helps identify data quality issues related to incorrect or unknown language detection.
    - Useful for ensuring that multilingual models have adequate and appropriate representation from various languages.

    ### Limitations

    - Dependency on the accuracy of the language detection library, which may not be perfect.
    - Languages with similar structures or limited text length may be incorrectly classified.
    - The test returns "Unknown" for entries where language detection fails, which might mask underlying issues with
    certain languages or text formats.
    """
    if not dataset.text_column:
        raise ValueError(
            "Please set the `text_column` option when "
            "initializing your Dataset object to use this test"
        )

    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return "Unknown"

    languages = dataset.df[dataset.text_column].apply(detect_language)

    return (
        px.histogram(
            languages,
            x=languages,
            title="Language Distribution",
            labels={"x": "Language Codes"},
        ),
        RawData(detected_languages=languages, dataset=dataset.input_id),
    )
