# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import string
from dataclasses import dataclass

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
from nltk.corpus import stopwords

from ....vm_models import Figure, Metric, VMDataset


@dataclass
class TextDescription(Metric):
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

    name = "text_description"
    required_inputs = ["dataset"]
    default_params = {
        "unwanted_tokens": {
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
        "num_top_words": 3,
        "lang": "english",
    }
    tasks = ["text_classification", "text_summarization"]
    tags = ["nlp", "text_data", "visualization"]

    def general_text_metrics(self, df, text_column):
        results = []

        for text in df[text_column]:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            paragraphs = text.split("\n\n")

            total_words = len(words)
            total_sentences = len(sentences)
            avg_sentence_length = round(
                (
                    sum(len(sentence.split()) for sentence in sentences)
                    / total_sentences
                    if total_sentences
                    else 0
                ),
                1,
            )
            total_paragraphs = len(paragraphs)

            results.append(
                [total_words, total_sentences, avg_sentence_length, total_paragraphs]
            )

        return pd.DataFrame(
            results,
            columns=[
                "Total Words",
                "Total Sentences",
                "Avg Sentence Length",
                "Total Paragraphs",
            ],
        )

    def vocabulary_structure_metrics(
        self, df, text_column, unwanted_tokens, num_top_words, lang
    ):
        stop_words = set(word.lower() for word in stopwords.words(lang))
        unwanted_tokens = set(token.lower() for token in unwanted_tokens)

        results = []

        for text in df[text_column]:
            words = nltk.word_tokenize(text)

            filtered_words = [
                word
                for word in words
                if word.lower() not in stop_words
                and word.lower() not in unwanted_tokens
                and word not in string.punctuation
            ]

            total_unique_words = len(set(filtered_words))
            total_punctuations = sum(1 for word in words if word in string.punctuation)
            lexical_diversity = round(
                total_unique_words / len(filtered_words) if filtered_words else 0, 1
            )

            results.append([total_unique_words, total_punctuations, lexical_diversity])

        return pd.DataFrame(
            results,
            columns=["Total Unique Words", "Total Punctuations", "Lexical Diversity"],
        )

    # Wrapper function that combines the outputs
    def text_description_table(self, df, params):
        text_column = self.inputs.dataset.text_column
        unwanted_tokens = params["unwanted_tokens"]
        num_top_words = params["num_top_words"]
        lang = params["lang"]

        gen_metrics_df = self.general_text_metrics(df, text_column)
        vocab_metrics_df = self.vocabulary_structure_metrics(
            df, text_column, unwanted_tokens, num_top_words, lang
        )
        combined_df = pd.concat([gen_metrics_df, vocab_metrics_df], axis=1)

        return combined_df

    def run(self):
        # Enforce that text_column must be provided as part of the params
        if self.inputs.dataset.text_column is None:
            raise ValueError("A 'text_column' must be provided to run this test.")

        # Can only run this test if we have a Dataset object
        if not isinstance(self.inputs.dataset, VMDataset):
            raise ValueError("TextDescription requires a validmind Dataset object")

        # download nltk data
        nltk.download("punkt_tab", quiet=True)

        df_text_description = self.text_description_table(
            self.inputs.dataset.df, self.params
        )

        # Define the combinations you want to plot
        combinations_to_plot = [
            ("Total Words", "Total Sentences"),
            ("Total Words", "Total Unique Words"),
            ("Total Sentences", "Avg Sentence Length"),
            ("Total Unique Words", "Lexical Diversity"),
        ]
        params = {"combinations_to_plot": combinations_to_plot}
        figures = self.text_description_plots(df_text_description, params)

        return self.cache_results(
            figures=figures,
        )

    # Function to plot scatter plots for specified combinations using Plotly
    def text_description_plots(self, df, params):
        combinations_to_plot = params["combinations_to_plot"]
        figures = []
        # Create hist plots for each column
        for i, column in enumerate(df.columns):
            fig = px.histogram(df, x=column)
            fig.update_layout(bargap=0.2)
            # Generate a unique key for each histogram using the column name and index
            histogram_key = f"{self.name}_histogram_{column}_{i}"
            figures.append(Figure(for_object=self, key=histogram_key, figure=fig))

        for j, (metric1, metric2) in enumerate(combinations_to_plot):
            fig = px.scatter(
                df, x=metric1, y=metric2, title=f"Scatter Plot: {metric1} vs {metric2}"
            )
            # Generate a unique key for each scatter plot using the metric names and index
            scatter_key = f"{self.name}_scatter_{metric1}_vs_{metric2}_{j}"
            figures.append(Figure(for_object=self, key=scatter_key, figure=fig))
        plt.close("all")

        return figures
