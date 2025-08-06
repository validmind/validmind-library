# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import warnings
from typing import Dict, Tuple

import plotly.express as px
import plotly.graph_objects as go
from datasets import Dataset

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset

from .utils import get_ragas_config, get_renamed_columns

try:
    from ragas import evaluate
    from ragas.metrics import LLMContextPrecisionWithoutReference as context_precision
except ImportError as e:
    if "ragas" in str(e):
        raise MissingDependencyError(
            "Missing required package `ragas` for ContextPrecision. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["ragas"],
            extra="llm",
        ) from e

    raise e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization", "text_classification")
def ContextPrecisionWithoutReference(
    dataset: VMDataset,
    user_input_column: str = "user_input",
    retrieved_contexts_column: str = "retrieved_contexts",
    response_column: str = "response",
    judge_llm=None,
    judge_embeddings=None,
) -> Tuple[Dict[str, list], go.Figure, go.Figure, RawData]:
    """
    Context Precision Without Reference is a metric used to evaluate the relevance of
    retrieved contexts compared to the expected response for a given user input. This
    metric compares each retrieved context (or chunk) with the response to estimate
    if the retrieved context is relevant.

    This metric can be used when you have both retrieved contexts and associated
    reference contexts for a `user_input`. Using a Language Model (LLM), it determines
    the relevance of each retrieved context by comparing it directly with the response,
    producing scores between 0 and 1, where higher scores indicate better precision in
    retrieving relevant contexts.

    ### Configuring Columns

    This metric requires the following columns in your dataset:

    - `user_input` (str): The user query or input to the model.
    - `retrieved_contexts` (List[str]): A list of text contexts retrieved for the
      user input that will be evaluated for relevance.
    - `response` (str): The model’s output response associated with the user input.

    If your dataset stores this data in different columns, you can specify alternate
    column names using the parameters `user_input_column`, `retrieved_contexts_column`,
    and `response_column`.

    Example configuration for custom column names:
    ```python
    {
        "user_input_column": "user_query",
        "retrieved_contexts_column": "retrieved_texts",
        "response_column": "model_output",
    }
    ```

    For datasets with data stored as dictionaries in other columns, specify the
    column and key like so:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": f"{pred_col}.contexts",
        "response_column": f"{pred_col}.response",
    }
    ```

    Alternatively, for complex situations, you may use a function to extract data:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": lambda x: [x[pred_col]["context_message"]],
        "response_column": "my_response_col",
    }
    ```
    """

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="promote has been superseded by promote_options='default'.",
    )

    required_columns = {
        "user_input": user_input_column,
        "retrieved_contexts": retrieved_contexts_column,
        "response": response_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)

    result_df = evaluate(
        Dataset.from_pandas(df),
        metrics=[context_precision()],
        **get_ragas_config(judge_llm, judge_embeddings)
    ).to_pandas()

    score_column = "llm_context_precision_without_reference"

    fig_histogram = px.histogram(
        x=result_df[score_column].to_list(), nbins=10, title="Context Precision"
    )
    fig_box = px.box(x=result_df[score_column].to_list(), title="Context Precision")

    return (
        {
            # "Scores (will not be uploaded to ValidMind Platform)": result_df[
            #     ["user_input", "retrieved_contexts", "response", "llm_context_precision_without_reference"]
            # ],
            "Aggregate Scores": [
                {
                    "Mean Score": result_df[score_column].mean(),
                    "Median Score": result_df[score_column].median(),
                    "Max Score": result_df[score_column].max(),
                    "Min Score": result_df[score_column].min(),
                    "Standard Deviation": result_df[score_column].std(),
                    "Count": result_df.shape[0],
                }
            ],
        },
        fig_histogram,
        fig_box,
        RawData(evaluation_results=result_df, dataset=dataset.input_id),
    )
