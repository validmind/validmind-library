"""Script that generates a description for a test using GPT-4 and automatically inserts it into the class docstring


Usage:
    poetry run python scripts/add_test_description.py <action> <path>

 - path: path to a test file or directory containing test files
 - action: `add` or `review`

Before running this, you need to either set an environment variable OPENAI_API_KEY
or create a .env file in the root of the project with the following contents:
OPENAI_API_KEY=<your api key>
"""

import os

import click
import dotenv
import openai

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_GPT_MODEL = "gpt-4o"  # or gpt-4-turbo or gpt-3.5-turbo etc


add_prompt = """
You are an expert in validating Machine Learning models using MRM (Model Risk Management) best practices.
You are also an expert in writing descriptions that are pleasant to read while being very useful.
You will be provided the source code for a test that is run against an ML model.
You will analyze the code to determine the details and implementation of the test.
Finally, you will write clear, descriptive and informative descriptions in the format described
below that will document the tests for developers and risk management teams.

For each test you will return a description with the following format and sections:
<Single sentence summary of the test>

<Short paragraph describing the test (should add more detail than the first sentence)>

1. Purpose
2. Test Mechanism
3. Signs of High Risk
4. Strengths
5. Limitations

Example description for a "Feature Drift" test:

```
Evaluates changes in feature distribution over time to identify potential model drift.

**Purpose**:
The Feature Drift test aims to evaluate how much the distribution of features has shifted over time between two datasets, typically training and monitoring datasets. It uses the Population Stability Index (PSI) to quantify this change, providing insights into the model’s robustness and the necessity for retraining or feature engineering.

**Test Mechanism**:
This test calculates the PSI by:

- Bucketing the distributions of each feature in both datasets.
- Comparing the percentage of observations in each bucket between the two datasets.
- Aggregating the differences across all buckets for each feature to produce the PSI score for that feature.

The PSI score is interpreted as:

- PSI < 0.1: No significant population change.
- PSI < 0.2: Moderate population change.
- PSI >= 0.2: Significant population change.

**Signs of High Risk**:
- PSI >= 0.2 for any feature, indicating a significant distribution shift.
- Consistently high PSI scores across multiple features.
- Sudden spikes in PSI in recent monitoring data compared to historical data.

**Strengths**:
- Provides a quantitative measure of feature distribution changes.
- Easily interpretable thresholds for decision-making.
- Helps in early detection of data drift, prompting timely interventions.

**Limitations**:
- May not capture more intricate changes in data distribution nuances.
- Assumes that bucket thresholds (quantiles) adequately represent distribution shifts.
- PSI score interpretation can be overly simplistic for complex datasets.
```

You will populate each section according to the following guidelines:
1. Purpose: Brief explanation of why this test is being used and what it is intended to evaluate or measure in relation to the model.
2. Test Mechanism: Describe the methodology used to test or apply the test, including any grading scales or thresholds
3. Signs of High Risk: List or describe any signs or indicators that might suggest a high risk or a failure in the model's performance as related to this metric
4. Strengths: List or describe the strengths or advantages of using this test in evaluating the model
5. Limitations: List or describe the limitations or disadvantages of this test, including any potential bias or areas it might not fully address

Ensure that each section is populated with succinct, clear, and relevant information pertaining to the test.
Respond with a markdown description where each section name is in header 3 format and then the content for that section.
Make sure to also remove the colon from the end of the header 3 section names and add a line break in between the section name and the section content.
For sections 1-2, make sure the content is in full sentences and paragraph form.
For sections 3-5, the content should be a list of bullet points unless the section has only one or two items, in which case it can be a paragraph.
Respond only with the description and don't include any explanation or other text. Additionally, avoid using enclosing markdown syntax like ```markdown
""".strip()


def is_test_function_signature(line, previous_line):
    """
    Test functions should have a @tags or @tasks decorator call on top of them
    """
    return line.startswith("def") and line.split("def ")[1].isupper()


def get_description_lines(lines):
    """
    Find the line number of the docstring that contains the description
    """
    # insert the description into the test code
    # the description should be inserted after the class definition line
    class_definition_line = None
    existing_description_lines = []

    advance_to_next_line = False
    for i, line in enumerate(lines):
        if advance_to_next_line or is_test_function_signature(line, lines[i - 1]):
            # ensure this is not a multi-line function signature like this:
            #
            # def test_function(
            #     arg1,
            #     arg2
            # ):
            #
            # we want to keep iterating until we find the closing parenthesis
            if ")" not in line:
                advance_to_next_line = True
                continue

            class_definition_line = i
            # check if there is already a doc string for the class
            if '"""' in lines[i + 1]:
                existing_description_lines.append(i + 1)
                j = i + 2
                while j < len(lines):
                    existing_description_lines.append(j)
                    if '"""' in lines[j]:
                        break
                    j += 1

            advance_to_next_line = False
            break

    if class_definition_line is None:
        raise ValueError("Could not find class or function definition line")

    return class_definition_line, existing_description_lines


def indent_and_wrap(text, indentation=4, wrap_length=120):
    lines = text.split("\n")
    result = []

    for line in lines:
        if line == "":
            result.append("")
            continue

        line = " " * indentation + line

        while len(line) > wrap_length:
            space_index = line.rfind(" ", 0, wrap_length)

            if space_index == -1:
                space_index = wrap_length

            result.append(line[:space_index])
            line = " " * indentation + line[space_index:].lstrip()

        result.append(line)

    return "\n".join(result)


def add_description_to_test(path):
    """Generate a test description using gpt4
    You can switch to gpt3.5 if you don't have access but gpt4 should do a better job
    """
    # get file contents from path
    click.echo(f"\n\n{path}:\n")
    with open(path, "r") as f:
        file_contents = f.read()

    response = openai.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=[
            {"role": "system", "content": add_prompt},
            {"role": "user", "content": f"```python\n{file_contents}```"},
        ],
        stream=True,
    )
    description = ""
    for chunk in response:
        if chunk.choices[0].finish_reason == "stop":
            break

        click.echo(chunk.choices[0].delta.content, nl=False)
        description += chunk.choices[0].delta.content

    click.echo("\n")

    # format the description to go into the test code
    # the description should be trimmed and have 4 spaces prepended to each line
    # each line should be wrapped at 120 characters
    description = indent_and_wrap(description.strip())
    lines = file_contents.split("\n")

    class_definition_line, existing_description_lines = get_description_lines(lines)

    # remove any existing description lines
    for i in reversed(existing_description_lines):
        lines.pop(i)

    # insert the new description lines
    lines.insert(class_definition_line + 1, f'    """\n{description}\n    """')

    # write the updated file contents back to the file
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _is_test_file(path):
    return path.endswith(".py") and path.split("/")[-1][0].isupper()


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def main(action, path):
    """Recursively processes the specified DIRECTORY and updates files needing metadata injection."""
    tests_to_process = []

    # check if path is a file or directory
    if os.path.isfile(path):
        if _is_test_file(path):
            tests_to_process.append(path)
        else:
            raise ValueError(f"File {path} is not a test file")

    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if _is_test_file(file):
                    tests_to_process.append(os.path.join(root, file))

    for file in tests_to_process:
        add_description_to_test(file)


if __name__ == "__main__":
    main()
