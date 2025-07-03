# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

from validmind import RawData, tags, tasks
from validmind.errors import MissingRequiredTestInputError

from .ai_powered_test import (
    call_model,
    get_explanation,
    get_score,
    missing_prompt_message,
)

SYSTEM = """
You are a prompt evaluation AI. You are aware of all prompt engineering best practices and can score prompts based on how well they satisfy different metrics. You analyse the prompts step-by-step based on provided documentation and provide a score and an explanation for how you produced that score.

Consider the following documentation on prompt clarity guidelines when evaluating the prompt:
'''
Clear prompts minimize the room for misinterpretation, allowing the LLM to generate more relevant and accurate responses. Ambiguous or vague instructions might leave the model guessing, leading to suboptimal outputs.

Tactics for Ensuring Clarity that will be referenced during evaluation:
1. Detail Inclusion: Provide essential details or context to prevent the LLM from making assumptions.
2. Adopt a Persona: Use system messages to specify the desired persona for the LLM's responses.
3. Specify Steps: For certain tasks, delineate the required steps explicitly, helping the model in sequential understanding.
4. Provide Examples: While general instructions are efficient, in some scenarios, "few-shot" prompting or style examples can guide the LLM more effectively.
5. Determine Output Length: Define the targeted length of the response, whether in terms of paragraphs, bullet points, or other units. While word counts aren't always precise, specifying formats like paragraphs can offer more predictable results.
'''

Score the clarity of the user-submitted prompt. Return a score from 1 to 10 where 10 is a perfect score. Also provide a short explanation for your score.

Response Format:
```
Score: <score>
Explanation: <explanation>
```
"""

USER = """
Prompt:
'''
{prompt_to_test}
'''
"""


@tags("llm", "zero_shot", "few_shot")
@tasks("text_classification", "text_summarization")
def Clarity(
    model, min_threshold=7, judge_llm=None
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Evaluates and scores the clarity of prompts in a Large Language Model based on specified guidelines.

    ### Purpose

    The Clarity evaluation metric is used to assess how clear the prompts of a Large Language Model (LLM) are. This
    assessment is particularly important because clear prompts assist the LLM in more accurately interpreting and
    responding to instructions.

    ### Test Mechanism

    The evaluation uses an LLM to scrutinize the clarity of prompts, factoring in considerations such as the inclusion
    of relevant details, persona adoption, step-by-step instructions, usage of examples, and specification of desired
    output length. Each prompt is rated on a clarity scale of 1 to 10, and any prompt scoring at or above the preset
    threshold (default of 7) will be marked as clear. It is important to note that this threshold can be adjusted via
    test parameters, providing flexibility in the evaluation process.

    ### Signs of High Risk

    - Prompts that consistently score below the clarity threshold
    - Repeated failure of prompts to adhere to guidelines for clarity, including detail inclusion, persona adoption,
    explicit step-by-step instructions, use of examples, and specification of output length

    ### Strengths

    - Encourages the development of more effective prompts that aid the LLM in interpreting instructions accurately
    - Applies a quantifiable measure (a score from 1 to 10) to evaluate the clarity of prompts
    - Threshold for clarity is adjustable, allowing for flexible evaluation depending on the context

    ### Limitations

    - Scoring system is subjective and relies on the AI’s interpretation of 'clarity'
    - The test assumes that all required factors (detail inclusion, persona adoption, step-by-step instructions, use of
    examples, and specification of output length) contribute equally to clarity, which might not always be the case
    - The evaluation may not be as effective if used on non-textual models
    """
    if not hasattr(model, "prompt"):
        raise MissingRequiredTestInputError(missing_prompt_message)

    response = call_model(
        system_prompt=SYSTEM,
        user_prompt=USER.format(prompt_to_test=model.prompt.template),
        judge_llm=judge_llm,
    )

    score = get_score(response)
    explanation = get_explanation(response)

    passed = score > min_threshold

    return (
        [
            {
                "Score": score,
                "Explanation": explanation,
                "Threshold": min_threshold,
                "Pass/Fail": "Pass" if passed else "Fail",
            }
        ],
        passed,
        RawData(response=response, model=model.input_id),
    )
