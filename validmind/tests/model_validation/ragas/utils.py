# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from langchain_core.outputs import ChatGeneration

from validmind.ai.utils import get_judge_config

EMBEDDINGS_MODEL = "text-embedding-3-small"


def _ragas_is_finished_parser(response) -> bool:
    """Accept finish signals used by Gemini as well as OpenAI-style responses."""

    is_finished_list = []
    accepted_finish_reasons = {
        "stop",
        "STOP",
        "end_turn",
        "END_TURN",
        "max_tokens",
        "MAX_TOKENS",
    }

    for generation_group in response.flatten():
        response_generation = generation_group.generations[0][0]
        generation_info = getattr(response_generation, "generation_info", None) or {}

        finish_reason = generation_info.get("finish_reason")
        if finish_reason is not None:
            is_finished_list.append(finish_reason in accepted_finish_reasons)
            continue

        if isinstance(response_generation, ChatGeneration):
            response_metadata = (
                getattr(response_generation.message, "response_metadata", None) or {}
            )
            finish_reason = response_metadata.get("finish_reason")
            stop_reason = response_metadata.get("stop_reason")

            if finish_reason is not None:
                is_finished_list.append(finish_reason in accepted_finish_reasons)
                continue

            if stop_reason is not None:
                is_finished_list.append(stop_reason in accepted_finish_reasons)
                continue

        is_finished_list.append(True)

    return all(is_finished_list)


def get_ragas_config(judge_llm=None, judge_embeddings=None):
    judge_llm, judge_embeddings = get_judge_config(judge_llm, judge_embeddings)

    try:
        from ragas.llms.base import LangchainLLMWrapper
    except ImportError:
        raise ImportError("Please run `pip install validmind[llm]` to use RAGAS tests")

    return {
        "llm": LangchainLLMWrapper(
            judge_llm, is_finished_parser=_ragas_is_finished_parser
        ),
        "embeddings": judge_embeddings,
    }


def make_sub_col_udf(root_col, sub_col):
    """Create a udf that extracts sub-column values from a dictionary."""

    def _udf_get_sub_col(x):
        if not isinstance(x, dict):
            raise TypeError(
                f"Expected a dictionary in column '{root_col}', got {type(x)}."
            )

        if sub_col not in x:
            raise KeyError(
                f"Sub-column '{sub_col}' not found in dictionary in column '{root_col}'."
            )

        return x[sub_col]

    return _udf_get_sub_col


def get_renamed_columns(df, column_map):
    """Get a new df with columns renamed according to the column_map

    Supports sub-column notation for getting values out of dictionaries that may be
    stored in a column. Also supports

    Args:
        df (pd.DataFrame): The DataFrame to rename columns in.
        column_map (dict): A dictionary mapping where the keys are the new column names
        that ragas expects and the values are one of the following:
            - The column name in the input dataframe
            - A string in the format "root_col.sub_col" to get a sub-column from a dictionary
            stored in a column.
            - A function that takes the value of the column and returns the value to be
            stored in the new column.

    Returns:
        pd.DataFrame: The DataFrame with columns renamed.
    """

    new_df = df.copy()

    for new_name, source in column_map.items():
        if callable(source):
            try:
                new_df[new_name] = new_df.apply(source, axis=1)
            except Exception as e:
                raise ValueError(
                    f"Failed to apply function to DataFrame. Error: {str(e)}"
                )

        elif "." in source:
            root_col, sub_col = source.split(".")

            if root_col in new_df.columns:
                new_df[new_name] = new_df[root_col].apply(
                    make_sub_col_udf(root_col, sub_col)
                )

            else:
                raise KeyError(f"Column '{root_col}' not found in DataFrame.")

        else:
            if source in new_df.columns:
                new_df[new_name] = new_df[source]

            else:
                raise KeyError(f"Column '{source}' not found in DataFrame.")

    return new_df
