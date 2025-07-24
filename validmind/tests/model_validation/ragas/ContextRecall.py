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
    from ragas.metrics import LLMContextRecall as context_recall
except ImportError as e:
    if "ragas" in str(e):
        raise MissingDependencyError(
            "Missing required package `ragas` for ContextRecall. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["ragas"],
            extra="llm",
        ) from e

    raise e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization", "text_classification")
def ContextRecall(
    dataset: VMDataset,
    user_input_column: str = "user_input",
    retrieved_contexts_column: str = "retrieved_contexts",
    reference_column: str = "reference",
    judge_llm=None,
    judge_embeddings=None,
) -> Tuple[Dict[str, list], go.Figure, go.Figure, RawData]:
    """
    Context recall measures the extent to which the retrieved context aligns with the
    annotated answer, treated as the ground truth. It is computed based on the `ground
    truth` and the `retrieved context`, and the values range between 0 and 1, with higher
    values indicating better performance.

    To estimate context recall from the ground truth answer, each sentence in the ground
    truth answer is analyzed to determine whether it can be attributed to the retrieved
    context or not. In an ideal scenario, all sentences in the ground truth answer
    should be attributable to the retrieved context.


    The formula for calculating context recall is as follows:
    $$
    \\text{context recall} = {|\\text{GT sentences that can be attributed to context}| \\over |\\text{Number of sentences in GT}|}
    $$

    ### Configuring Columns

    This metric requires the following columns in your dataset:

    - `user_input` (str): The text query that was input into the model.
    - `retrieved_contexts` (List[str]): A list of text contexts which are retrieved and
    which will be evaluated to make sure they contain all items in the ground truth.
    - `reference` (str): The ground truth text to compare with the retrieved contexts.

    If the above data is not in the appropriate column, you can specify different column
    names for these fields using the parameters `user_input_column`,
    `retrieved_contexts_column` and `reference_column`.

    For example, if your dataset has this data stored in different columns, you can
    pass the following parameters:
    ```python
    {
        "user_input_column": "user_input",
        "retrieved_contexts_column": "retrieved_contexts",
        "reference_column": "reference",
    }
    ```

    If the data is stored as a dictionary in another column, specify the column and key
    like this:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": f"{pred_col}.retrieved_contexts",
        "reference_column": f"{pred_col}.reference",
    }
    ```

    For more complex situations, you can use a function to extract the data:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": lambda x: [x[pred_col]["retrieved_contexts"]],
        "reference_column": "my_ground_truth_col",
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
        "reference": reference_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)
    df = df[required_columns.keys()]

    result_df = evaluate(
        Dataset.from_pandas(df),
        metrics=[context_recall()],
        **get_ragas_config(judge_llm, judge_embeddings)
    ).to_pandas()

    score_column = "context_recall"

    fig_histogram = px.histogram(
        x=result_df[score_column].to_list(), nbins=10, title="Context Recall"
    )
    fig_box = px.box(x=result_df[score_column].to_list(), title="Context Recall")

    return (
        {
            # "Scores (will not be uploaded to ValidMind Platform)": result_df[
            #     ["question", "contexts", "ground_truth", "context_recall"]
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
