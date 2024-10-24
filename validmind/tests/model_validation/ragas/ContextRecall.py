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
    from ragas.metrics import context_recall
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `ragas` for ContextRecall. "
        "Please run `pip install validmind[llm]` to use LLM tests",
        required_dependencies=["ragas"],
        extra="llm",
    ) from e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization", "text_classification")
def ContextRecall(
    dataset,
    question_column: str = "question",
    contexts_column: str = "contexts",
    ground_truth_column: str = "ground_truth",
):
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

    - `question` (str): The text query that was input into the model.
    - `contexts` (List[str]): A list of text contexts which are retrieved and which
    will be evaluated to make sure they contain all items in the ground truth.
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
        Dataset.from_pandas(df), metrics=[context_recall], **get_ragas_config()
    ).to_pandas()

    fig_histogram = px.histogram(x=result_df["context_recall"].to_list(), nbins=10)
    fig_box = px.box(x=result_df["context_recall"].to_list())

    return (
        {
            # "Scores (will not be uploaded to UI)": result_df[
            #     ["question", "contexts", "ground_truth", "context_recall"]
            # ],
            "Aggregate Scores": [
                {
                    "Mean Score": result_df["context_recall"].mean(),
                    "Median Score": result_df["context_recall"].median(),
                    "Max Score": result_df["context_recall"].max(),
                    "Min Score": result_df["context_recall"].min(),
                    "Standard Deviation": result_df["context_recall"].std(),
                    "Count": result_df.shape[0],
                }
            ],
        },
        fig_histogram,
        fig_box,
    )
