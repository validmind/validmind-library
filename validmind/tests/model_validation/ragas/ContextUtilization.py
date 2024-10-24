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
    from ragas.metrics import context_utilization
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `ragas` for ContextUtilization. "
        "Please run `pip install validmind[llm]` to use LLM tests",
        required_dependencies=["ragas"],
        extra="llm",
    ) from e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization", "text_classification")
def ContextUtilization(
    dataset,
    question_column: str = "question",
    contexts_column: str = "contexts",
    answer_column: str = "answer",
):  # noqa: B950
    """
    Assesses how effectively relevant context chunks are utilized in generating answers by evaluating their ranking
    within the provided contexts.

    ### Purpose

    The Context Utilization test evaluates whether all of the answer-relevant items present in the contexts are ranked
    higher within the provided retrieval results. This metric is essential for assessing the performance of models,
    especially those involved in tasks such as text QA, text generation, text summarization, and text classification.

    ### Test Mechanism

    The test calculates Context Utilization using the formula:

    $$
    \\text{Context Utilization@K} = \\frac{\\sum_{k=1}^{K} \\left( \\text{Precision@k} \\times v_k \\right)}{\\text{Total number of relevant items in the top } K \\text{ results}}
    $$
    $$
    \\text{Precision@k} = {\\text{true positives@k} \\over  (\\text{true positives@k} + \\text{false positives@k})}
    $$

    Where $K$ is the total number of chunks in `contexts` and $v_k \\in \\{0, 1\\}$ is the relevance indicator at rank $k$.


    This test uses columns for questions, contexts, and answers from the dataset and computes context utilization
    scores, generating a histogram and box plot for visualization.

    #### Configuring Columns

    This metric requires the following columns in your dataset:

    - `question` (str): The text query that was input into the model.
    - `contexts` (List[str]): A list of text contexts which are retrieved and which will be evaluated to
       make sure they contain relevant info in the correct order.
    - `answer` (str): The llm-generated response for the input `question`.

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

    ### Signs of High Risk

    - Very low mean or median context utilization scores, indicating poor usage of retrieved contexts.
    - High standard deviation, suggesting inconsistent model performance.
    - Low or minimal max scores, pointing to the model's failure to rank relevant contexts at top positions.

    ### Strengths

    - Quantifies the rank of relevant context chunks in generating responses.
    - Provides clear visualizations through histograms and box plots for ease of interpretation.
    - Adapts to different dataset schema by allowing configurable column names.

    ### Limitations

    - Assumes the relevance of context chunks is binary and may not capture nuances of partial relevance.
    - Requires proper context retrieval to be effective; irrelevant context chunks can skew the results.
    - Dependent on large sample sizes to provide stable and reliable estimates of utilization performance.
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="promote has been superseded by promote_options='default'.",
    )

    required_columns = {
        "question": question_column,
        "contexts": contexts_column,
        "answer": answer_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)

    result_df = evaluate(
        Dataset.from_pandas(df), metrics=[context_utilization], **get_ragas_config()
    ).to_pandas()

    fig_histogram = px.histogram(x=result_df["context_utilization"].to_list(), nbins=10)
    fig_box = px.box(x=result_df["context_utilization"].to_list())

    return (
        {
            # "Scores (will not be uploaded to UI)": result_df[
            #     ["question", "contexts", "answer", "context_utilization"]
            # ],
            "Aggregate Scores": [
                {
                    "Mean Score": result_df["context_utilization"].mean(),
                    "Median Score": result_df["context_utilization"].median(),
                    "Max Score": result_df["context_utilization"].max(),
                    "Min Score": result_df["context_utilization"].min(),
                    "Standard Deviation": result_df["context_utilization"].std(),
                    "Count": result_df.shape[0],
                }
            ],
        },
        fig_histogram,
        fig_box,
    )
