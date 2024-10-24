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
    from ragas.metrics import context_entity_recall
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `ragas` for ContextEntityRecall. "
        "Please run `pip install validmind[llm]` to use LLM tests",
        required_dependencies=["ragas"],
        extra="llm",
    ) from e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization")
def ContextEntityRecall(
    dataset,
    contexts_column: str = "contexts",
    ground_truth_column: str = "ground_truth",
):
    """
    Evaluates the context entity recall for dataset entries and visualizes the results.

    ### Overview

    This metric gives the measure of recall of the retrieved context, based on the
    number of entities present in both `ground_truths` and `contexts` relative to the
    number of entities present in the `ground_truths` alone. Simply put, it is a measure
    of what fraction of entities are recalled from `ground_truths`. This metric is
    useful in fact-based use cases like tourism help desk, historical QA, etc. This
    metric can help evaluate the retrieval mechanism for entities, based on comparison
    with entities present in `ground_truths`, because in cases where entities matter,
    we need the `contexts` which cover them.

    ### Formula

    To compute this metric, we use two sets, $GE$ and $CE$, representing the set of
    entities present in `ground_truths` and set of entities present in `contexts`
    respectively. We then take the number of elements in intersection of these sets and
    divide it by the number of elements present in the $GE$, given by the formula:

    $$
    \\text{context entity recall} = \\frac{| CE \\cap GE |}{| GE |}
    $$

    ### Configuring Columns

    This metric requires the following columns in your dataset:

    - `contexts` (List[str]): A list of text contexts which will be evaluated to make
    sure if they contain the entities present in the ground truth.
    - `ground_truth` (str): The ground truth text from which the entities will be
    extracted and compared with the entities in the `contexts`.

    If the above data is not in the appropriate column, you can specify different column
    names for these fields using the parameters `contexts_column`, and `ground_truth_column`.

    For example, if your dataset has this data stored in different columns, you can
    pass the following parameters:
    ```python
    {
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
        "contexts_column": lambda row: [row[pred_col]["context_message"]],
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
        "ground_truth": ground_truth_column,
        "contexts": contexts_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)

    result_df = evaluate(
        Dataset.from_pandas(df), metrics=[context_entity_recall], **get_ragas_config()
    ).to_pandas()

    fig_histogram = px.histogram(
        x=result_df["context_entity_recall"].to_list(), nbins=10
    )
    fig_box = px.box(x=result_df["context_entity_recall"].to_list())

    return (
        {
            # "Scores (will not be uploaded to UI)": result_df[
            #     [
            #         "contexts",
            #         "ground_truth",
            #         "context_entity_recall",
            #     ]
            # ],
            "Aggregate Scores": [
                {
                    "Mean Score": result_df["context_entity_recall"].mean(),
                    "Median Score": result_df["context_entity_recall"].median(),
                    "Max Score": result_df["context_entity_recall"].max(),
                    "Min Score": result_df["context_entity_recall"].min(),
                    "Standard Deviation": result_df["context_entity_recall"].std(),
                    "Count": result_df.shape[0],
                }
            ],
        },
        fig_histogram,
        fig_box,
    )
