"""Script that runs bulk updates on test files using AI


Usage:
    poetry run python scripts/bulk_ai_test_updates.py <path> --action <action>

 - path: path to a test file or directory containing test files
 - action: `add_description` or `add_raw_data`

Before running this, you need to either set an environment variable OPENAI_API_KEY
or create a .env file in the root of the project with the following contents:
OPENAI_API_KEY=<your api key>
"""

import os

import click
import dotenv
from openai import OpenAI
from pydantic import BaseModel
from textwrap import indent, fill

dotenv.load_dotenv()

OPENAI_GPT_MODEL = "gpt-4o"  # or gpt-4-turbo or gpt-3.5-turbo etc

client = OpenAI()


class TestDescription(BaseModel):
    summary: str
    purpose: str
    test_mechanism: str
    signs_of_high_risk: list[str]
    strengths: list[str]
    limitations: list[str]

    def to_str(self):
        def list_to_str(lst):
            my_str = ""
            for item in lst:
                my_str += indent(fill(f" - {item}", width=116), "    ")
                my_str += "\n"
            return my_str.strip("\n")

        # formatted to 120 chars wide and indented 4 spaces as its a function docstring
        return f'''    """
{indent(fill(self.summary, width=116), "    ")}

    **Purpose**:

{indent(fill(self.purpose, width=116), "    ")}

    **Test Mechanism**:

{indent(fill(self.test_mechanism, width=116), "    ")}

    **Signs of High Risk**:
{list_to_str(self.signs_of_high_risk)}

    **Strengths**:
{list_to_str(self.strengths)}

    **Limitations**:
{list_to_str(self.limitations)}
    """'''


add_prompt = """
You are an expert in validating Machine Learning models using MRM (Model Risk Management) best practices.
You are also an expert in writing descriptions that are pleasant to read while being very useful.
You will be provided the source code for a test that is run against an ML model.
You will analyze the code to determine the details and implementation of the test.
Finally, you will write clear, descriptive and informative descriptions in the format described below that will document the test.

Ignore existing docstrings if you think they are incorrect or incomplete. The code itself should be the source of truth.

For each test you will write and return the following sections:

1. Short single sentence summary of the test
2. Purpose
3. Test Mechanism
4. Signs of High Risk
5. Strengths
6. Limitations

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
1. Summary: A single sentence summary of the test that is easy to digest for both technical and non-technical audiences.
2. Purpose: Brief explanation of why this test is being used and what it is intended to evaluate or measure in relation to the model.
3. Test Mechanism: Describe the methodology used to test or apply the test, including any grading scales or thresholds
4. Signs of High Risk: Short list of the signs or indicators that might suggest a high risk or a failure in the model's performance as related to this metric
5. Strengths: Short list of the strengths or advantages of using this test in evaluating the model
6. Limitations: Short list of the limitations or disadvantages of this test, including any potential bias or areas it might not fully address

Ensure that each section is populated with succinct, clear, and relevant information pertaining to the test.
For sections 1-3, make sure the content is in full sentences and paragraph form.
For sections 4-6, the content should be a list of bullet points returned as a list of strings. Keep the list short and concise and only include the most important points.
""".strip()


def add_description_to_test(path):
    """Generate a test description using gpt4
    You can switch to gpt3.5 if you don't have access but gpt4 should do a better job
    """
    # get file contents from path
    click.echo(f"> {path}")
    with open(path, "r") as f:
        file_contents = f.read()

    test_name = path.split("/")[-1].split(".")[0]

    response = client.beta.chat.completions.parse(
        model=OPENAI_GPT_MODEL,
        messages=[
            {"role": "system", "content": add_prompt},
            {"role": "user", "content": f"```python\n{file_contents}```"},
        ],
        response_format=TestDescription,
    )
    description = response.choices[0].message.parsed

    lines = file_contents.split("\n")

    # find the test function definition
    test_def_start = 0
    test_def_end = 0
    # find start of test function definition
    for i, line in enumerate(lines):
        if "def" in line and test_name in line:
            test_def_start = i
            break
    # handle multiline test function definitions
    for i, line in enumerate(lines[test_def_start:]):
        if "):" in line:
            test_def_end = i + test_def_start + 1
            break
    # handle existing docstrings
    if '"""' in lines[test_def_end]:
        lines_to_remove = [test_def_end]
        for i, line in enumerate(lines[test_def_end + 1 :]):
            lines_to_remove.append(test_def_end + i + 1)
            if '"""' in line:
                break
        for i in reversed(lines_to_remove):
            lines.pop(i)
    # insert the new description lines
    lines.insert(test_def_end, description.to_str())

    with open(path, "w") as f:
        f.write("\n".join(lines))


def add_raw_data_to_test(path):
    """Add raw data to a test file"""
    pass


def _is_test_file(path):
    return path.endswith(".py") and path.split("/")[-1][0].isupper()


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
)
@click.option(
    "--action",
    type=click.Choice(["add_description", "add_raw_data"]),
    required=True,
)
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

    if action == "add_description":
        func = add_description_to_test
    elif action == "add_raw_data":
        func = add_raw_data_to_test
    else:
        raise ValueError(f"Invalid action: {action}")

    for file in tests_to_process:
        func(file)


if __name__ == "__main__":
    main()
