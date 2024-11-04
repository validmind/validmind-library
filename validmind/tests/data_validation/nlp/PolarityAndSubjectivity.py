# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


import pandas as pd
import plotly.express as px
from textblob import TextBlob

from validmind import tags, tasks


@tags("nlp", "text_data", "data_validation")
@tasks("nlp")
def PolarityAndSubjectivity(dataset):
    """
    Analyzes the polarity and subjectivity of text data within a given dataset to visualize the sentiment distribution.

    ### Purpose

    The Polarity and Subjectivity test is designed to evaluate the sentiment expressed in textual data. By analyzing
    these aspects, it helps to identify the emotional tone and subjectivity of the dataset, which could be crucial in
    understanding customer feedback, social media sentiments, or other text-related data.

    ### Test Mechanism

    This test uses TextBlob to compute the polarity and subjectivity scores of textual data in a given dataset. The
    mechanism includes:

    - Iterating through each text entry in the specified column of the dataset.
    - Applying the TextBlob library to compute the polarity (ranging from -1 for negative sentiment to +1 for positive
    sentiment) and subjectivity (ranging from 0 for objective to 1 for subjective) for each entry.
    - Creating a scatter plot using Plotly to visualize the relationship between polarity and subjectivity.

    ### Signs of High Risk

    - High concentration of negative polarity values indicating prevalent negative sentiments.
    - High subjectivity scores suggesting the text data is largely opinion-based rather than factual.
    - Disproportionate clusters of extreme scores (e.g., many points near -1 or +1 polarity).

    ### Strengths

    - Quantifies sentiment and subjectivity which can provide actionable insights.
    - Visualizes sentiment distribution, aiding in easy interpretation.
    - Utilizes well-established TextBlob library for sentiment analysis.

    ### Limitations

    - Polarity and subjectivity calculations may oversimplify nuanced text sentiments.
    - Reliance on TextBlob which may not be accurate for all domains or contexts.
    - Visualization could become cluttered with very large datasets, making interpretation difficult.
    """
    sentiments = dataset.df[dataset.text_column].apply(lambda x: TextBlob(x).sentiment)

    return px.scatter(
        data_frame=pd.DataFrame(
            {
                "polarity": [s.polarity for s in sentiments],
                "subjectivity": [s.subjectivity for s in sentiments],
            }
        ),
        x="polarity",
        y="subjectivity",
        title="Polarity vs Subjectivity",
        labels={
            "polarity": "Polarity (-1 negative to +1 positive)",
            "subjectivity": "Subjectivity (0 objective to 1 subjective)",
        },
    )
