# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Agent interface for all text generation tasks
"""

import requests

from validmind.api_client import _get_api_headers, _get_url, raise_api_error
from validmind.vm_models.result import TestResult


def run_task(generation_type: str, input: dict) -> TestResult:
    """
    Run text generation for different purposes like code explanation.

    Args:
        generation_type (str): Type of text generation ('code_explainer' or 'qualitative_text')
        input (dict): Dictionary containing source_code and parameters

    Returns:
        TestResult: Test result object containing the generated text
    """
    if generation_type == "code_explainer":
        r = requests.post(
            url=_get_url("ai/generate/code_explainer"),
            headers=_get_api_headers(),
            json=input,
        )

        if r.status_code != 200:
            raise_api_error(r.text)

        generated_text = r.json()["content"]
    else:
        raise ValueError(f"Unsupported generation type: {generation_type}")

    # Create a test result with the generated text
    result = TestResult(
        result_id=f"{generation_type}",
        description=generated_text,
        title=f"Text Generation: {generation_type}",
        doc=f"Generated {generation_type}",
    )

    return result
