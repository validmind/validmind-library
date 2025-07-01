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
You are a prompt evaluation AI. You are aware of all prompt engineering best practices and can score prompts based on how well they satisfy different best practices. You analyse the prompts step-by-step based on provided documentation and provide a score and an explanation for how you produced that score.

Consider the following best practice documentation:
```
Its been shown that, when passing examples to an LLM via the prompt for the purpose of Sentiment Analysis, a distribution with a majority of positive examples led the LLM to classify ambiguous sentences as "Positive", and vice versa. However, for more straightforward tasks, such as sentiment classification, biases may be harder to induce. The order of presentation also appeared to influence the LLM's outputs, especially when combined with a skewed distribution of exemplars.
Tactics for Avoiding Bias in Prompting:
1. Balanced Distribution: Avoid heavily skewed distributions of examples. Instead, offer amore balanced number of examples for each label.
2. Randomize Order: To prevent order-induced biases, shuffle the sequence of examples ensuring no fixed pattern, such as presenting all positive examples first.
3. Continued Experimentation: Regularly test and tweak prompts, especially for complex tasks where the LLM might not have extensive knowledge. This iterative process helps in identifying potential pitfalls and refining prompting strategies.
Example:
For instance, using 8 positive and 2 negative examples led the LLM to classify an ambiguous sentence "I feel something" as "Positive". However, with a more balanced distribution, the LLM showed unbiased behavior.
```

Score the bias of the user-submitted prompt. Return a score from 1 to 10 where 10 is a perfect score. Also provide a short explanation for your score.

Response Format:
```
Score: <score>
Explanation: <explanation>
```
""".strip()

USER = '''
Prompt:
"""
{prompt_to_test}
"""
'''.strip()


@tags("llm", "few_shot")
@tasks("text_classification", "text_summarization")
def Bias(
    model, min_threshold=7, judge_llm=None
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Assesses potential bias in a Large Language Model by analyzing the distribution and order of exemplars in the
    prompt.

    ### Purpose

    The Bias Evaluation test calculates if and how the order and distribution of exemplars (examples) in a few-shot
    learning prompt affect the output of a Large Language Model (LLM). The results of this evaluation can be used to
    fine-tune the model's performance and manage any unintended biases in its results.

    ### Test Mechanism

    This test uses two checks:

    1. **Distribution of Exemplars:** The number of positive vs. negative examples in a prompt is varied. The test then
    examines the LLM's classification of a neutral or ambiguous statement under these circumstances.
    2. **Order of Exemplars:** The sequence in which positive and negative examples are presented to the model is
    modified. Their resultant effect on the LLM's response is studied.

    For each test case, the LLM grades the input prompt on a scale of 1 to 10. It evaluates whether the examples in the
    prompt could produce biased responses. The test only passes if the score meets or exceeds a predetermined minimum
    threshold. This threshold is set at 7 by default but can be modified as per the requirements via the test
    parameters.

    ### Signs of High Risk

    - A skewed result favoring either positive or negative responses may suggest potential bias in the model. This skew
    could be caused by an unbalanced distribution of positive and negative exemplars.
    - If the score given by the model is less than the set minimum threshold, it might indicate a risk of high bias and
    hence poor performance.

    ### Strengths

    - This test provides a quantitative measure of potential bias, offering clear guidelines for developers about
    whether their Large Language Model (LLM) contains significant bias.
    - It is useful in evaluating the impartiality of the model based on the distribution and sequence of examples.
    - The flexibility to adjust the minimum required threshold allows tailoring this test to stricter or more lenient
    bias standards.

    ### Limitations

    - The test may not pick up on more subtle forms of bias or biases that are not directly related to the distribution
    or order of exemplars.
    - The test's effectiveness will decrease if the quality or balance of positive and negative exemplars is not
    representative of the problem space the model is intended to solve.
    - The use of a grading mechanism to gauge bias may not be entirely accurate in every case, particularly when the
    difference between threshold and score is narrow.
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
        RawData(response=response),
    )
