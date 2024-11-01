# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import warnings

import plotly.express as px
from datasets import Dataset

from validmind import tags, tasks
from validmind.errors import MissingDependencyError

from .utils import get_ragas_config, get_renamed_columns

try:
    from ragas import evaluate
    from ragas.metrics import context_precision
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `ragas` for ContextPrecision. "
        "Please run `pip install validmind[llm]` to use LLM tests",
        required_dependencies=["ragas"],
        extra="llm",
    ) from e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization", "text_classification")
def ContextPrecision(
    dataset,
    question_column: str = "question",
    contexts_column: str = "contexts",
    ground_truth_column: str = "ground_truth",
):  # noqa: B950
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

    - `question` (str): The text query that was input into the model.
    - `contexts` (List[str]): A list of text contexts which are retrieved and which
    will be evaluated to make sure they contain relevant info in the correct order.
    - `ground_truth` (str): The ground truth text to compare with the retrieved contexts.

    If the above data is not in the appropriate column, you can specify different column
    names for these fields using the parameters `question_column`, `contexts_column`
    and `ground_truth_column`.

    For example, if your dataset has this data stored in different columns, you can
    pass the following parameters:
    ```python
    {
        "question_column": "question",
        "contexts_column": "context_info"
        "ground_truth_column": "my_ground_truth_col",
    }
    ```

    If the data is stored as a dictionary in another column, specify the column and key
    like this:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "contexts_column": f"{pred_col}.contexts",
        "ground_truth_column": "my_ground_truth_col",
    }
    ```

    For more complex situations, you can use a function to extract the data:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "contexts_column": lambda x: [x[pred_col]["context_message"]],
        "ground_truth_column": "my_ground_truth_col",
    }
    ```
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="promote has been superseded by promote_options='default'.",
    )

    required_columns = {
        "question": question_column,
        "contexts": contexts_column,
        "ground_truth": ground_truth_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)

    result_df = evaluate(
        Dataset.from_pandas(df), metrics=[context_precision], **get_ragas_config()
    ).to_pandas()

    fig_histogram = px.histogram(x=result_df["context_precision"].to_list(), nbins=10)
    fig_box = px.box(x=result_df["context_precision"].to_list())

    return (
        {
            # "Scores (will not be uploaded to UI)": result_df[
            #     ["question", "contexts", "ground_truth", "context_precision"]
            # ],
            "Aggregate Scores": [
                {
                    "Mean Score": result_df["context_precision"].mean(),
                    "Median Score": result_df["context_precision"].median(),
                    "Max Score": result_df["context_precision"].max(),
                    "Min Score": result_df["context_precision"].min(),
                    "Standard Deviation": result_df["context_precision"].std(),
                    "Count": result_df.shape[0],
                }
            ],
        },
        fig_histogram,
        fig_box,
    )
