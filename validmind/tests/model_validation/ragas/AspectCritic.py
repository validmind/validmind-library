# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import warnings
from typing import Dict, List, Optional, Tuple

import plotly.express as px
import plotly.graph_objects as go
from datasets import Dataset

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset

from .utils import get_ragas_config, get_renamed_columns

try:
    from ragas import evaluate
    from ragas.metrics import AspectCritic as aspect_critic
    from ragas.metrics._aspect_critic import (
        coherence,
        conciseness,
        correctness,
        harmfulness,
        maliciousness,
    )
except ImportError as e:
    if "ragas" in str(e):
        raise MissingDependencyError(
            "Missing required package `ragas` for AspectCritic. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["ragas"],
            extra="llm",
        ) from e

    raise e

LOWER_IS_BETTER_ASPECTS = ["harmfulness", "maliciousness"]


@tags("ragas", "llm", "qualitative")
@tasks("text_summarization", "text_generation", "text_qa")
def AspectCritic(
    dataset: VMDataset,
    user_input_column: str = "user_input",
    response_column: str = "response",
    retrieved_contexts_column: Optional[str] = None,
    aspects: List[str] = [
        "coherence",
        "conciseness",
        "correctness",
        "harmfulness",
        "maliciousness",
    ],
    additional_aspects: Optional[List[Tuple[str, str]]] = None,
    judge_llm=None,
    judge_embeddings=None,
) -> Tuple[Dict[str, list], go.Figure, RawData]:
    """
    Evaluates generations against the following aspects: harmfulness, maliciousness,
    coherence, correctness, and conciseness.

    ### Overview:

    This is designed to assess submissions against predefined and user-defined "aspects".
    For each aspect, a judge LLM is prompted to critique a piece of generated text based
    on a description of the aspect. The output of this evaluation is a binary (0/1 = yes/no)
    score that indicates whether the submission aligns with the defined aspect or not.

    ### Inputs and Outputs:

    The input to this metric is a dataset containing the input `user_input` (prompt to the LLM)
    and the `response` (text generated by the LLM). Any retrieved `retrieved_contexts` can also be
    included to enhance the evaluation.

    The `user_input_column`, `response_column`, and `retrieved_contexts_column` parameters can be used to
    specify the names or sources for the data that this metric will evaluate if the dataset
    does not contain the required columns `user_input`, `response`, and `retrieved_contexts`.

    By default, the aspects evaluated are harmfulness, maliciousness, coherence,
    correctness, and conciseness. To change the aspects evaluated, the `aspects` parameter
    can be set to a list containing any of these aspects.

    To add custom aspects, the `additional_aspects` parameter can be passed as a list
    of tuples where each tuple contains the aspect name and a description of the aspect
    that the judge LLM will use to critique the submission.

    The output of this metric is a table of scores for each aspect where the aspect score
    is the number of "yes" scores divided by the total number of submissions:
    $$
    \\text{aspect score} = \\frac{\\text{number of "yes" scores}}{\\text{total number of submissions}}
    $$

    ### Examples:

    - **Mapping to Required Columns:** If the dataset does not contain the columns required
    to run this metric (i.e., `user_input`, `response`, and `retrieved_contexts`), the

    ```python
    pred_col = my_vm_dataset.prediction_column(my_vm_model)
    run_test(
        "validmind.model_validation.ragas.AspectCritic",
        inputs={"dataset": my_vm_dataset},
        params={
            "user_input_column": "input_prompt",
            "response_column": f"{pred_col}.llm_output",
            "retrieved_contexts_column": "retrieval_model_prediction",
        },
    )
    ```

    - **Custom Aspects:** To evaluate custom aspects, the `additional_aspects` parameter can
    be set to a list of tuples where each tuple contains the aspect name and a description
    of the aspect that the judge LLM will use to critique the submission. For example, to
    evaluate whether the LLM-generated text has a "professional tone", the `additional_aspects`
    parameter can be set like this:

    ```python
    run_test(
        "validmind.model_validation.ragas.AspectCritic",
        inputs={"dataset": my_vm_dataset},
        params={
            "additional_aspects": [
                ("professionalism", "Does the text have a professional tone?"),
            ],
        },
    )
    ```
    """
    built_in_aspects = {
        "coherence": coherence,
        "conciseness": conciseness,
        "correctness": correctness,
        "harmfulness": harmfulness,
        "maliciousness": maliciousness,
    }

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="promote has been superseded by promote_options='default'.",
    )

    required_columns = {
        "user_input": user_input_column,
        "response": response_column,
    }

    if retrieved_contexts_column:
        required_columns["retrieved_contexts"] = retrieved_contexts_column
    df = get_renamed_columns(dataset._df, required_columns)
    df = df[required_columns.keys()]

    custom_aspects = (
        [
            aspect_critic(name=name, definition=description)
            for name, description in additional_aspects
        ]
        if additional_aspects
        else []
    )
    all_aspects = [built_in_aspects[aspect] for aspect in aspects] + custom_aspects

    result_df = evaluate(
        Dataset.from_pandas(df),
        metrics=all_aspects,
        **get_ragas_config(judge_llm, judge_embeddings)
    ).to_pandas()

    # reverse the score for aspects where lower is better
    for aspect in LOWER_IS_BETTER_ASPECTS:
        if aspect in result_df.columns:
            result_df[aspect] = 1 - result_df[aspect]

    df_melted = result_df.melt(
        id_vars=["user_input", "response"]
        + (["retrieved_contexts"] if retrieved_contexts_column else []),
        value_vars=[aspect.name for aspect in all_aspects],
        var_name="Metric",
        value_name="Result",
    )
    df_counts = df_melted.groupby(["Metric", "Result"]).size().reset_index(name="Count")
    df_counts["Result"] = df_counts["Result"].map({0: "Fail", 1: "Pass"})

    fig = px.bar(
        df_counts,
        x="Metric",
        y="Count",
        color="Result",
        color_discrete_map={"Fail": "red", "Pass": "green"},
        labels={"Count": "Pass vs Fail Count", "Metric": "Aspect Name"},
        barmode="group",
        title="Aspect Critique Results",
    )

    return (
        {
            "Aspect Scores": [
                {"Aspect": aspect, "Score": result_df[aspect].mean()}
                for aspect in aspects + [aspect.name for aspect in custom_aspects]
            ]
        },
        fig,
        RawData(
            evaluation_results=result_df,
            dataset=dataset.input_id,
        ),
    )
