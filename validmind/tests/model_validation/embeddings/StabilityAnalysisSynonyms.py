# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import random
from typing import Tuple

import nltk
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import wordnet as wn

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from .utils import create_stability_analysis_result


@tags("llm", "text_data", "embeddings", "visualization")
@tasks("feature_extraction")
def StabilityAnalysisSynonyms(
    dataset: VMDataset,
    model: VMModel,
    probability: float = 0.02,
    mean_similarity_threshold: float = 0.7,
) -> Tuple[go.Figure, pd.DataFrame, RawData]:
    """
    Evaluates the stability of text embeddings models when words in test data are replaced by their synonyms randomly.

    ### Purpose

    The Stability Analysis Synonyms test is designed to gauge the robustness and stability of an embeddings model on
    text-based data. The test does so by introducing random word changes through replacing words in the test dataset
    with their synonyms.

    ### Test Mechanism

    This test utilizes WordNet to find synonyms for a given word present in the test data, replacing the original word
    with this synonym based on a given probability. The probability is defined as a parameter and determines the
    likelihood of swapping a word with its synonym. By default, this is set at 0.02 but can be adjusted based on
    specific test requirements. This methodology enables an evaluation of how such replacements can affect the model's
    performance.

    ### Signs of High Risk

    - The model's performance or predictions change significantly after swapping words with their synonyms.
    - The model shows high sensitivity to small perturbations, like modifying the data with synonyms.
    - The embeddings model fails to identify similar meanings between the original words and their synonyms, indicating
    it lacks semantic understanding.

    ### Strengths

    - The test is flexible in its application. The 'probability' parameter can be adjusted based on the degree of
    synonym swapping required.
    - Efficient in gauging a model's sensitivity or robustness with respect to small changes in input data.
    - Provides insights into the semantic understanding of the model as it monitors the impact of swapping words with
    synonyms.

    ### Limitations

    - The ability to perturb data is reliant on the availability of synonyms, limiting its efficiency.
    - It assumes that the synonyms provided by WordNet are accurate and interchangeable in all contexts, which may not
    always be the case given the intricacies of language and context-specific meanings.
    - It does not consider the influence of multi-word expressions or phrases, as synonyms are considered at the word
    level only.
    - Relies solely on the WordNet corpus for synonyms, limiting its effectiveness for specialized or domain-specific
    jargon not included in that corpus.
    - Does not consider the semantic role of the words in the sentence, meaning the swapped synonym could potentially
    alter the overall meaning of the sentence, leading to a false perception of the model's stability.
    """
    # download the nltk wordnet
    nltk.download("wordnet", quiet=True)

    def perturb_data(data):
        if not isinstance(data, str):
            return data

        words = nltk.word_tokenize(data)
        modified_words = []

        # For each word, check the probability and swap if needed
        for word in words:
            if random.random() <= probability:
                # get synonyms for the word
                synonyms = [
                    lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas()
                ]
                # filter out original word
                synonyms = [syn for syn in synonyms if syn != word]

                if synonyms:
                    modified_word = random.choice(synonyms)
                    # WordNet uses underscores for multi-word synonyms
                    modified_words.append(modified_word.replace("_", " "))
                    continue

            modified_words.append(word)

        return " ".join(modified_words)

    original_df = dataset.df[[dataset.text_column]]
    perturbed_df = original_df.copy()
    perturbed_df[dataset.text_column] = perturbed_df[dataset.text_column].map(
        perturb_data
    )

    raw_data, result = create_stability_analysis_result(
        dataset.y_pred(model),
        model.predict(perturbed_df),
        mean_similarity_threshold,
    )

    return *result, RawData(
        original_perturbed_similarity=raw_data,
        model=model.input_id,
        dataset=dataset.input_id,
    )
