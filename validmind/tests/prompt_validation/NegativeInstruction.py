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
You are a prompt evaluation AI.
You are aware of all prompt engineering best practices and can score prompts based on how well they satisfy different metrics.
You analyse the prompts step-by-step based on provided documentation and provide a score and an explanation for how you produced that score.

Consider the following documentation regarding negative instructions in prompts and utilize it to grade the user-submitted prompt:
'''
Best practices for LLM prompt engineering suggest that positive instructions should be preferred over negative instructions.
For example, instead of saying "Don't do X", it is better to say "Do Y".
This is because the model is more likely to generate the desired output if it is given a positive instruction.
Prompts that are phrased in the affirmative, emphasizing what to do, tend to direct the LLM more clearly than those that focus on what not to do.
Negative instructions can lead to ambiguities and undesired model responses.
By emphasizing clarity and proactive guidance, we optimize the chances of obtaining relevant and targeted responses from the LLM.

Example:
Consider a scenario involving a chatbot designed to recommend movies.
An instruction framed as, "Don't recommend movies that are horror or thriller", might cause the LLM to fixate on the genres mentioned,inadvertently producing undesired results.
On the other hand, a positively-framed prompt like, "Recommend family-friendly movies or romantic comedies" provides clear guidance on the desired output.
'''

Based on this best practice, please score the user-submitted prompt on a scale of 1-10, where 10 is a perfect score.
Provide an explanation for your score.

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
def NegativeInstruction(
    model, min_threshold=7, judge_llm=None
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Evaluates and grades the use of affirmative, proactive language over negative instructions in LLM prompts.

    ### Purpose

    The Negative Instruction test is utilized to scrutinize the prompts given to a Large Language Model (LLM). The
    objective is to ensure these prompts are expressed using proactive, affirmative language. The focus is on
    instructions indicating what needs to be done rather than what needs to be avoided, thereby guiding the LLM more
    efficiently towards the desired output.

    ### Test Mechanism

    An LLM is employed to evaluate each prompt. The prompt is graded based on its use of positive instructions with
    scores ranging between 1-10. This grade reflects how effectively the prompt leverages affirmative language while
    shying away from negative or restrictive instructions. A prompt that attains a grade equal to or above a
    predetermined threshold (7 by default) is regarded as adhering effectively to the best practices of positive
    instruction. This threshold can be custom-tailored through the test parameters.

    ### Signs of High Risk

    - Low score obtained from the LLM analysis, indicating heavy reliance on negative instructions in the prompts.
    - Failure to surpass the preset minimum threshold.
    - The LLM generates ambiguous or undesirable outputs as a consequence of the negative instructions used in the
    prompt.

    ### Strengths

    - Encourages the usage of affirmative, proactive language in prompts, aiding in more accurate and advantageous
    model responses.
    - The test result provides a comprehensible score, helping to understand how well a prompt follows the positive
    instruction best practices.

    ### Limitations

    - Despite an adequate score, a prompt could still be misleading or could lead to undesired responses due to factors
    not covered by this test.
    - The test necessitates an LLM for evaluation, which might not be available or feasible in certain scenarios.
    - A numeric scoring system, while straightforward, may oversimplify complex issues related to prompt designing and
    instruction clarity.
    - The effectiveness of the test hinges significantly on the predetermined threshold level, which can be subjective
    and may need to be adjusted according to specific use-cases.
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
                "Threshold": min_threshold,
                "Explanation": explanation,
                "Pass/Fail": "Pass" if passed else "Fail",
            }
        ],
        passed,
        RawData(model_response=response, model=model.input_id),
    )
