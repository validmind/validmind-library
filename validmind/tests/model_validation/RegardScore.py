# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import evaluate
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.tests.utils import validate_prediction
from validmind.vm_models import VMDataset, VMModel


@tags("nlp", "text_data", "visualization")
@tasks("text_classification", "text_summarization")
def RegardScore(
    dataset: VMDataset,
    model: VMModel,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Assesses the sentiment and potential biases in text generated by NLP models by computing and visualizing regard
    scores.

    ### Purpose

    The `RegardScore` test aims to evaluate the levels of regard (positive, negative, neutral, or other) in texts
    generated by NLP models. It helps in understanding the sentiment and bias present in the generated content.

    ### Test Mechanism

    This test extracts the true and predicted values from the provided dataset and model. It then computes the regard
    scores for each text instance using a preloaded `regard` evaluation tool. The scores are compiled into dataframes,
    and visualizations such as histograms and bar charts are generated to display the distribution of regard scores.
    Additionally, descriptive statistics (mean, median, standard deviation, minimum, and maximum) are calculated for
    the regard scores, providing a comprehensive overview of the model's performance.

    ### Signs of High Risk

    - Noticeable skewness in the histogram, especially when comparing the predicted regard scores with the target
    regard scores, can indicate biases or inconsistencies in the model.
    - Lack of neutral scores in the model's predictions, despite a balanced distribution in the target data, might
    signal an issue.

    ### Strengths

    - Provides a clear evaluation of regard levels in generated texts, aiding in ensuring content appropriateness.
    - Visual representations (histograms and bar charts) make it easier to interpret the distribution and trends of
    regard scores.
    - Descriptive statistics offer a concise summary of the model's performance in generating texts with balanced
    sentiments.

    ### Limitations

    - The accuracy of the regard scores is contingent upon the underlying `regard` tool.
    - The scores provide a broad overview but do not specify which portions or tokens of the text are responsible for
    high regard.
    - Supplementary, in-depth analysis might be needed for granular insights.
    """

    # Extract true and predicted values
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Ensure equal lengths and get truncated data if necessary
    y_true, y_pred = validate_prediction(y_true, y_pred)

    # Load the regard evaluation metric
    regard_tool = evaluate.load("regard", module_type="measurement")

    # Function to calculate regard scores
    def compute_regard_scores(texts):
        scores = regard_tool.compute(data=texts)["regard"]
        regard_dicts = [
            dict((x["label"], x["score"]) for x in sublist) for sublist in scores
        ]
        return regard_dicts

    # Calculate regard scores for true and predicted texts
    true_regard = compute_regard_scores(y_true)
    pred_regard = compute_regard_scores(y_pred)

    # Convert scores to dataframes
    true_df = pd.DataFrame(true_regard)
    pred_df = pd.DataFrame(pred_regard)

    figures = []

    # Function to create histogram and bar chart for regard scores
    def create_figures(df, title):
        for category in df.columns:
            # Histogram
            hist_fig = go.Figure(data=[go.Histogram(x=df[category])])
            hist_fig.update_layout(
                title=f"{title} - {category.capitalize()} Histogram",
                xaxis_title=category.capitalize(),
                yaxis_title="Count",
            )
            figures.append(hist_fig)

            # Bar Chart
            bar_fig = go.Figure(data=[go.Bar(x=df.index, y=df[category])])
            bar_fig.update_layout(
                title=f"{title} - {category.capitalize()} Bar Chart",
                xaxis_title="Text Instance Index",
                yaxis_title=category.capitalize(),
            )
            figures.append(bar_fig)

    # Create figures for each regard score dataframe
    create_figures(true_df, "True Text Regard")
    create_figures(pred_df, "Predicted Text Regard")

    # Calculate statistics for each regard score dataframe
    def calculate_stats(df, metric_name):
        stats = df.describe().loc[["mean", "50%", "max", "min", "std"]].T
        stats.columns = [
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
        ]
        stats["Metric"] = metric_name
        stats["Count"] = len(df)
        return stats

    true_stats = calculate_stats(true_df, "True Text Regard")
    pred_stats = calculate_stats(pred_df, "Predicted Text Regard")

    # Combine statistics into a single dataframe
    result_df = (
        pd.concat([true_stats, pred_stats])
        .reset_index()
        .rename(columns={"index": "Category"})
    )
    result_df = result_df[
        [
            "Metric",
            "Category",
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
    ]

    return (
        result_df,
        *figures,
        RawData(
            true_regard=true_df,
            pred_regard=pred_df,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
