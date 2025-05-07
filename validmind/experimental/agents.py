# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Agent interface for all text generation tasks
"""

import requests

from validmind.api_client import _get_api_headers, _get_url, raise_api_error
from validmind.utils import is_html, md_to_html
from validmind.vm_models.result import TextGenerationResult


def run_task(
    task: str,
    input: dict,
    show: bool = True,
) -> TextGenerationResult:
    """
    Run text generation tasks using AI models.

    Args:
        task (str): Type of text generation task to run. Currently supports:
            - 'code_explainer': Generates natural language explanations of code
        input (dict): Input parameters for the generation task:
            - For code_explainer: Must contain 'source_code' and optional parameters
        show (bool): Whether to display the generated result. Defaults to True.

    Returns:
        TextGenerationResult: Result object containing the generated text and metadata

    Raises:
        ValueError: If an unsupported task is provided
        requests.exceptions.RequestException: If the API request fails
    """
    if task == "code_explainer" or task == "qualitative_text_generation":
        r = requests.post(
            url=_get_url(f"ai/generate/{task}"),
            headers=_get_api_headers(),
            json=input,
        )

        if r.status_code != 200:
            raise_api_error(r.text)

        generated_text = r.json()["content"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    if not is_html(generated_text):
        generated_text = md_to_html(generated_text, mathml=True)

    # Create a test result with the generated text
    result = TextGenerationResult(
        result_type=f"{task}",
        description=generated_text,
        title=f"Text Generation: {task}",
        doc=f"Generated {task}",
    )
    if show:
        result.show()

    return result
