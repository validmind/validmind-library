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
import subprocess
import textwrap

import click
import dotenv
from openai import OpenAI
from pydantic import BaseModel
from textwrap import indent, fill

dotenv.load_dotenv()

OPENAI_GPT_MODEL = "gpt-4o"  # or gpt-4-turbo or gpt-3.5-turbo etc

client = OpenAI()

USER_PROMPT = (
    None  # can be hardcoded instead of it being prompted from the command line
)
# USER_PROMPT = """
# Can you change any tests that add a bunch of figures to a list called `returns` and then add a raw data or other stuff to the list at the end before converting it to a tuple in the return.
# These instances would be cleaner to read if instead a `figures` list was used and this list was unpacked into the return tuple like this:
# ```
# returns = []
# ...
# returns.append(RawData(some_key=some_value))
# return tuple(returns)
# ```

# to this:

# ```
# figures = []
# ...
# return (*figures, RawData(some_key=some_value))
# ```

# If the test doesn't have any figures or only has one figure,or the figures are not stored in a list called `returns`, then just return `NO CHANGE`.
# You are looking for the exact pattern described above. Avoid unnecessary changes.
# """


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

The Feature Drift test aims to evaluate how much the distribution of features has shifted over time between two datasets, typically training and monitoring datasets. It uses the Population Stability Index (PSI) to quantify this change, providing insights into the model's robustness and the necessity for retraining or feature engineering.

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


raw_data_prompt = """
You are an expert Python engineer and data scientist with broad experience across many domains.
ValidMind is a company that provides a Python SDK for building and running tests for the purposes of model risk management.
ValidMind's SDK offers a library of "test" functions that are run with our test harness against many types of models and datasets.
These test functions return either a single object or a tuple of objects.
These objects are turned into a test result report by the test harness.
They can return any number of the following types of objects:
- Tables (dataframes or lists of dictionaries)
- Figures (matplotlib or plotly figures)
- Values (scalars, vectors, etc.)
- Pass/Fail (a boolean value that indicates whether the test passed or failed)
- Raw Data (intermediate data helpful for re-generating any of the above objects when post-processing the test result)

Tests can return either a single object from the above list or a tuple of these objects in any order.

The raw data is a new feature that allows tests to return intermediate data that is not appropriate to show in the test result but is helpful if the user adds a post-processing function to modify the test result.
Its a class that can be initialized with any number of any type of objects using a key-value like interface where the key in the constructor is the name of the object and the value is the object itself.
It should only be used to store data that is not already returned as part of the test result (i.e. in a table) but could be useful to re-generate any of the test result objects (tables, figures).

When adding raw data, you should always include:
- If the test has access to a model parameter (VMModel), include its input_id as model=model.input_id
- If the test has access to a dataset parameter (VMDataset), include its input_id as dataset=dataset.input_id
Only include these if they are available in the test function parameters - don't force both if only one is accessible.

You will be provided with the source code for a "test" that is run against an ML model or dataset.
You will analyze the code to determine the details and implementation of the test.
Then you will use the below example to implement changes to the test to make it use the new raw data mechanism offered by the ValidMind SDK.

Example test without raw data:

```
... # existing code, imports, etc.
from validmind import tags, tasks
...

def ExampleConfusionMatrix(model: VMModel, dataset: VMDataset):
    y_pred = dataset.y_pred(model)
    y_true = dataset.y.astype(y_pred.dtype)

    labels = np.unique(y_true)
    labels = sorted(labels.tolist())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = ff.create_annotated_heatmap()
    ..

    return fig
```

Example test with raw data:

```
... # existing code, imports, etc.
from validmind import tags, tasks, RawData
...


def ExampleConfusionMatrix(model: VMModel, dataset: VMDataset):

    y_pred = dataset.y_pred(model)
    y_true = dataset.y.astype(y_pred.dtype)

    labels = np.unique(y_true)
    labels = sorted(labels.tolist())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = ff.create_annotated_heatmap()
    ..

    return fig, RawData(confusion_matrix=cm, model=model.input_id, dataset=dataset.input_id)
```

Notice that the test now returns a tuple of the figure and the raw data.
Also notice the import of the RawData object.

You will return the updated test code (make sure to include all the existing imports, copyrights, comments, etc.).
Return only the updated code and nothing else.
Do not wrap the code in backticks, simply return valid Python code.
If the test already uses the RawData object, simply return the original code without any changes and without backticks.

Prefer dataframes over dictionaries or numpy arrays when adding raw data but don't force it if the test only uses dictionaries or some other format.
Be intentional about the name of the key in the RawData object, it should be a short, descriptive name that is easy for developers to understand and use.
Do not use vague names like "data", "results", "output", etc. Use something specific to the test and descriptive of the data being stored.
Ideally, the raw data should end up containing anything needed to re-generate the final output (assuming that the original inputs and parameters are available).

If the test doesn't really have anything that should be stored as raw data, just return the original code and nothing else.
If the test already returns one or more tables that include all the data that you would add to the raw data, then don't add raw data.
The user will prefer the tables over the raw data since they are easier to understand and use.
The raw data is most impactful when it contains data used to produce one or more figures or when it contains intermediate data that is used to produce aggregated or summary tables.

Some notes:
- ValidMind tests should return a single tuple.
- Multiple figures, tables, etc. can be returned as part of a single top level return tuple.
- The only exception is when the test returns multiple tables with titles. These are returned as a dictionary where the keys are the table titles and the values are the tables objects (dataframes or lists of dictionaries).
- If the test uses a list to collect multiple figures, etc. and then converts that list to a tuple when returning, you should add the raw data to the end of the list before it is converted to a tuple.
    - In this case, rename the list if its "figures" or something similar to avoid confusion (a good name would be "returns")

DO NOT CHANGE ANYTHING OTHER THAN ADDING THE NEW RAW DATA MECHANISM... I.E. DO NOT REMOVE ANYTHING FROM THE RETURN TUPLE OR THE RETURN VALUE (if it is a single object)
"""

custom_prompt_system = """
You are an expert Python engineer and data scientist with broad experience across many domains.
ValidMind is a company that provides a Python SDK for building and running tests for the purposes of model risk management.
ValidMind's SDK offers a library of "test" functions that are run with our test harness against many types of models and datasets.
These test functions return either a single object or a tuple of objects.
These objects are turned into a test result report by the test harness.
They can return any number of the following types of objects:
- Tables (dataframes or lists of dictionaries)
- Figures (matplotlib or plotly figures)
- Values (scalars, vectors, etc.)
- Pass/Fail (a boolean value that indicates whether the test passed or failed)
- Raw Data (intermediate data helpful for re-generating any of the above objects when post-processing the test result)

Tests can return either a single object from the above list or a tuple of these objects in any order.

You will be provided with custom instructions from the user on how to modify one or more tests.

You will then be provided with the source code for a test (one by one).
You will analyze the code carefully and then generate an updated version of the code to meet the user's instructions.
You will then return the updated test code (make sure to include all the existing imports, copyrights, comments, etc.).

Return only the updated code and nothing else.
Do not wrap the code in backticks, simply return valid Python code.

The only execption is if the test doesn't need to be modified.
In this case, return the following string exactly: `NO CHANGE`
"""

review_prompt_system = """
You are an expert Python engineer and data scientist with broad experience across many domains.
ValidMind is a company that provides a Python SDK for building and running tests for the purposes of model risk management.
ValidMind's SDK offers a library of "test" functions that are run with our test harness against many types of models and datasets.
These test functions return either a single object or a tuple of objects.
These objects are turned into a test result report by the test harness.
They can return any number of the following types of objects:
- Tables (dataframes or lists of dictionaries)
- Figures (matplotlib or plotly figures)
- Values (scalars, vectors, etc.)
- Pass/Fail (a boolean value that indicates whether the test passed or failed)
- Raw Data (intermediate data helpful for re-generating any of the above objects when post-processing the test result)

Tests can return either a single object from the above list or a tuple of these objects in any order.

You will be provided with custom instructions from the user on how to review one or more tests.

You will then be provided with the source code for a test (one by one).
You will analyze the code to determine the details and implementation of the test.
You will follow the user's instructions to review and then provide feedback on the test.

If the test does not need any feedback, simply return the following string exactly: `NO FEEDBACK NEEDED`
"""


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
    # get file contents from path
    click.echo(f"> {path}")
    with open(path, "r") as f:
        file_contents = f.read()

    response = client.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=[
            {"role": "system", "content": raw_data_prompt},
            {"role": "user", "content": f"```python\n{file_contents}```"},
        ],
    )

    updated_file_contents = response.choices[0].message.content
    # remove starting "```python" and ending "```"
    updated_file_contents = (
        updated_file_contents.lstrip("```python").rstrip("```").strip()
    )

    with open(path, "w") as f:
        f.write(updated_file_contents)


def custom_prompt(path, user_prompt):
    """Custom prompt for a test file"""
    # get file contents from path
    click.echo(f"> {path}")
    with open(path, "r") as f:
        file_contents = f.read()

    response = client.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=[
            {"role": "system", "content": custom_prompt_system},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": f"```python\n{file_contents}```"},
        ],
    )

    if "NO CHANGE" in response.choices[0].message.content:
        click.echo("No changes needed")
        return

    updated_file_contents = response.choices[0].message.content
    # remove starting "```python" and ending "```"
    updated_file_contents = (
        updated_file_contents.lstrip("```python").rstrip("```").strip()
    )

    with open(path, "w") as f:
        f.write(updated_file_contents)


def custom_review(path, user_prompt):
    """Custom review for a test file"""
    # get file contents from path
    click.echo(f"\n> {path}")
    with open(path, "r") as f:
        file_contents = f.read()

    response = client.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=[
            {"role": "system", "content": review_prompt_system},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": f"```python\n{file_contents}```"},
        ],
    )

    if "NO FEEDBACK NEEDED" in response.choices[0].message.content:
        click.echo("No feedback needed")
        return

    feedback = response.choices[0].message.content
    click.echo(textwrap.indent(feedback, "    "))


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
    type=click.Choice(
        ["add_description", "add_raw_data", "custom_prompt", "custom_review"]
    ),
    required=True,
)
@click.option(
    "--model",
    type=click.Choice(["gpt-4o", "gpt-4o-mini"]),
    default="gpt-4o",
)
def main(action, path, model):
    """Recursively processes the specified DIRECTORY and updates files needing metadata injection."""
    global OPENAI_GPT_MODEL
    OPENAI_GPT_MODEL = model

    tests_to_process = []

    # check if path is a file or directory
    if os.path.isfile(path):
        if _is_test_file(path):
            tests_to_process.append(path)
        else:
            raise ValueError(f"File {path} is not a test file")

    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if _is_test_file(file):
                    tests_to_process.append(os.path.join(root, file))

    if action == "add_description":
        func = add_description_to_test
    elif action == "add_raw_data":
        func = add_raw_data_to_test
    elif action == "custom_prompt":
        if not USER_PROMPT:
            user_prompt = input("Enter your prompt: ")
            user_prompt = user_prompt.strip("\n").strip()
        else:
            user_prompt = USER_PROMPT
        func = lambda path: custom_prompt(path, user_prompt)
    elif action == "custom_review":
        review_prompt = input("Enter your review prompt: ")
        review_prompt = review_prompt.strip("\n").strip()
        func = lambda path: custom_review(path, review_prompt)
    else:
        raise ValueError(f"Invalid action: {action}")

    for file in tests_to_process:
        func(file)

    # run black on the tests directory
    subprocess.run(["poetry", "run", "black", "validmind/tests"])


if __name__ == "__main__":
    main()
