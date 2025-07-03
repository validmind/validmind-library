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
    from ragas.metrics import LLMContextPrecisionWithReference as context_precision
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
def ContextPrecision(
    dataset: VMDataset,
    user_input_column: str = "user_input",
    retrieved_contexts_column: str = "retrieved_contexts",
    reference_column: str = "reference",
    judge_llm=None,
    judge_embeddings=None,
) -> Tuple[Dict[str, list], go.Figure, go.Figure, RawData]:
    """
    Context Precision is a metric that evaluates whether all of the ground-truth
    relevant items present in the contexts are ranked higher or not. Ideally all the
    relevant chunks must appear at the top ranks. This metric is computed using the
    `question`, `ground_truth` and the `contexts`, with values ranging between 0 and 1,
    where higher scores indicate better precision.

    $$
    \\text{Context Precision@K} = \\frac{\\sum_{k=1}^{K} \\left( \\text{Precision@k} \\times v_k \\right)}{\\text{Total number of relevant items in the top } K \\text{ results}}
    $$
    $$
    \\text{Precision@k} = {\\text{true positives@k} \\over  (\\text{true positives@k} + \\text{false positives@k})}
    $$

    Where $K$ is the total number of chunks in contexts and $v_k \\in \\{0, 1\\}$ is the
    relevance indicator at rank $k$.

    ### Configuring Columns

    This metric requires the following columns in your dataset:

    - `user_input` (str): The text query that was input into the model.
    - `retrieved_contexts` (List[str]): A list of text contexts which are retrieved and which
    will be evaluated to make sure they contain relevant info in the correct order.
    - `reference` (str): The ground truth text to compare with the retrieved contexts.

    If the above data is not in the appropriate column, you can specify different column
    names for these fields using the parameters `user_input_column`, `retrieved_contexts_column`
    and `reference_column`.

    For example, if your dataset has this data stored in different columns, you can
    pass the following parameters:
    ```python
    {
        "user_input_column": "question",
        "retrieved_contexts_column": "context_info",
        "reference_column": "my_ground_truth_col",
    }
    ```

    If the data is stored as a dictionary in another column, specify the column and key
    like this:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": f"{pred_col}.retrieved_contexts",
        "reference_column": "my_ground_truth_col",
    }
    ```

    For more complex situations, you can use a function to extract the data:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": lambda x: [x[pred_col]["context_message"]],
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

    result_df = evaluate(
        Dataset.from_pandas(df),
        metrics=[context_precision()],
        **get_ragas_config(judge_llm, judge_embeddings)
    ).to_pandas()

    score_column = "llm_context_precision_with_reference"

    fig_histogram = px.histogram(
        x=result_df[score_column].to_list(), nbins=10, title="Context Precision"
    )
    fig_box = px.box(x=result_df[score_column].to_list(), title="Context Precision")

    return (
        {
            # "Scores (will not be uploaded to ValidMind Platform)": result_df[
            #     ["user_input", "retrieved_contexts", "reference", "llm_context_precision_with_reference"]
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
