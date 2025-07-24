# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import MissingRequiredTestInputError, SkipTestError

from .ai_powered_test import call_model, missing_prompt_message

SYSTEM = """
You are a prompt evaluation researcher AI who is tasked with testing the robustness of LLM prompts.

Consider the following guidelines:
'''
LLM prompts are used to guide a model to generate specific outputs or solve specific tasks.
These prompts can be more or less robust, meaning that they can be more or less susceptible to breaking especially when the output needs to be a specific type.
A robust prompt ensures consistent performance and reduces the likelihood of unexpected or off-tangent outputs.
This consistency is vital for applications where predictability and reliability of the LLM's response are paramount.
'''

Consider the user-submitted prompt template and generate an input for the variable in the template (denoted by brackets) that tests the robustness of the prompt.
Contradictions, edge cases, typos, bad phrasing, distracting, complex or out-of-place words and phrases are just some of the strategies you can use when generating inputs.
Be creative and think step-by-step how you would break the prompt.
Then generate {num_tests} inputs for the user-submitted prompt template that would break the prompt.
Each input should be different from the others.
Each input should be returned as a new line in your response.
Respond only with the values to be inserted into the prompt template and do not include quotes, explanations or any extra text.

Example:

User-provided prompt:
```
Analyse the following sentence and output its sentiment:
\\{sentence\\}
```

Your response (generated inputs):
```
I am a happy cat
You are a bad person
My name is bob
James' friend is really sick
```
"""

USER = """
Prompt:
```
{prompt_to_test}
```
Input:
"""


@tags("llm", "zero_shot", "few_shot")
@tasks("text_classification", "text_summarization")
def Robustness(
    model, dataset, num_tests=10, judge_llm=None
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Assesses the robustness of prompts provided to a Large Language Model under varying conditions and contexts. This test
    specifically measures the model's ability to generate correct classifications with the given prompt even when the
    inputs are edge cases or otherwise difficult to classify.

    ### Purpose

    The Robustness test is meant to evaluate the resilience and reliability of prompts provided to a Language Learning
    Model (LLM). The aim of this test is to guarantee that the prompts consistently generate accurate and expected
    outputs, even in diverse or challenging scenarios. This test is only applicable to LLM-powered text classification
    tasks where the prompt has a single input variable.

    ### Test Mechanism

    The Robustness test appraises prompts under various conditions, alterations, and contexts to ascertain their
    stability in producing consistent responses from the LLM. Factors evaluated include different phrasings, inclusion
    of potential distracting elements, and various input complexities. By default, the test generates 10 inputs for a
    prompt but can be adjusted according to test parameters.

    ### Signs of High Risk

    - If the output from the tests diverges extensively from the expected results, this indicates high risk.
    - When the prompt doesn't give a consistent performance across various tests.
    - A high risk is indicated when the prompt is susceptible to breaking, especially when the output is expected to be
    of a specific type.

    ### Strengths

    - The robustness test helps to ensure stable performance of the LLM prompts and lowers the chances of generating
    unexpected or off-target outputs.
    - This test is vital for applications where predictability and reliability of the LLM’s output are crucial.

    ### Limitations

    - Currently, the test only supports single-variable prompts, which restricts its application to more complex models.
    - When there are too many target classes (over 10), the test is skipped, which can leave potential vulnerabilities
    unchecked in complex multi-class models.
    - The test may not account for all potential conditions or alterations that could show up in practical use
    scenarios.
    """
    if not hasattr(model, "prompt"):
        raise MissingRequiredTestInputError(missing_prompt_message)

    if not model.prompt.variables or len(model.prompt.variables) > 1:
        raise SkipTestError("Robustness only supports single-variable prompts for now")

    target_class_labels = dataset.target_classes()
    if len(target_class_labels) > 10:
        raise SkipTestError(
            "Too many target classes to test robustness. Skipping test."
        )

    generated_inputs = call_model(
        system_prompt=SYSTEM.format(num_tests=num_tests),
        user_prompt=USER.format(prompt_to_test=model.prompt.template),
        judge_llm=judge_llm,
    ).split("\n")

    responses = model.predict(
        pd.DataFrame(generated_inputs, columns=model.prompt.variables)
    )

    results = [
        {
            "Generated Input": generated_input,
            "Model Response": response,
            "Pass/Fail": "Pass" if response in target_class_labels else "Fail",
        }
        for generated_input, response in zip(generated_inputs, responses)
    ]

    return (
        results,
        all(result["Pass/Fail"] == "Pass" for result in results),
        RawData(
            generated_inputs=generated_inputs,
            responses=responses,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
