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
    from ragas.metrics import ContextEntityRecall as context_entity_recall
except ImportError as e:
    if "ragas" in str(e):
        raise MissingDependencyError(
            "Missing required package `ragas` for ContextEntityRecall. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["ragas"],
            extra="llm",
        ) from e

    raise e


@tags("ragas", "llm", "retrieval_performance")
@tasks("text_qa", "text_generation", "text_summarization")
def ContextEntityRecall(
    dataset: VMDataset,
    retrieved_contexts_column: str = "retrieved_contexts",
    reference_column: str = "reference",
    judge_llm=None,
    judge_embeddings=None,
) -> Tuple[Dict[str, list], go.Figure, go.Figure, RawData]:
    """
    Evaluates the context entity recall for dataset entries and visualizes the results.

    ### Overview

    This metric gives the measure of recall of the retrieved context, based on the
    number of entities present in both `reference` and `retrieved_contexts` relative to the
    number of entities present in the `reference` alone. Simply put, it is a measure
    of what fraction of entities are recalled from `reference`. This metric is
    useful in fact-based use cases like tourism help desk, historical QA, etc. This
    metric can help evaluate the retrieval mechanism for entities, based on comparison
    with entities present in `reference`, because in cases where entities matter,
    we need the `retrieved_contexts` which cover them.

    ### Formula

    To compute this metric, we use two sets, $GE$ and $CE$, representing the set of
    entities present in `reference` and set of entities present in `retrieved_contexts`
    respectively. We then take the number of elements in intersection of these sets and
    divide it by the number of elements present in the $GE$, given by the formula:

    $$
    \\text{context entity recall} = \\frac{| CE \\cap GE |}{| GE |}
    $$

    ### Configuring Columns

    This metric requires the following columns in your dataset:

    - `retrieved_contexts` (List[str]): A list of text contexts which will be evaluated to make
    sure if they contain the entities present in the `reference`.
    - `reference` (str): The ground truth text from which the entities will be
    extracted and compared with the entities in the `retrieved_contexts`.

    If the above data is not in the appropriate column, you can specify different column
    names for these fields using the parameters `retrieved_contexts_column`, and `reference_column`.

    For example, if your dataset has this data stored in different columns, you can
    pass the following parameters:
    ```python
    {
        "retrieved_contexts_column": "context_info",
        "reference_column": "my_ground_truth_col",
    }
    ```

    If the data is stored as a dictionary in another column, specify the column and key
    like this:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": f"{pred_col}.contexts",
        "reference_column": "my_ground_truth_col",
    }
    ```

    For more complex situations, you can use a function to extract the data:
    ```python
    pred_col = dataset.prediction_column(model)
    params = {
        "retrieved_contexts_column": lambda row: [row[pred_col]["context_message"]],
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
        "reference": reference_column,
        "retrieved_contexts": retrieved_contexts_column,
    }

    df = get_renamed_columns(dataset._df, required_columns)

    result_df = evaluate(
        Dataset.from_pandas(df),
        metrics=[context_entity_recall()],
        **get_ragas_config(judge_llm, judge_embeddings)
    ).to_pandas()

    score_column = "context_entity_recall"

    fig_histogram = px.histogram(
        x=result_df[score_column].to_list(), nbins=10, title="Context Entity Recall"
    )
    fig_box = px.box(x=result_df[score_column].to_list(), title="Context Entity Recall")

    return (
        {
            # "Scores (will not be uploaded to ValidMind Platform)": result_df[
            #     [
            #         "retrieved_contexts",
            #         "reference",
            #         "context_entity_recall",
            #     ]
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
