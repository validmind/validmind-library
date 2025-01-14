"""This script updates all unit tests for the ValidMind tests

Ensure that the tests to be updated are working properly since this will overwrite the existing unit tests
to expect whatever is returned from the test as the source of truth.

To just update the unit tests if there have been changes to the tests, run with the --update-only flag.

To create new unit tests and update existing unit tests, run without the --update-only flag.

Example:
```bash
# create a new unit test for a test called UniqueValues
python scripts/bulk_unit_tests_updates.py validmind/tests/data_validation/UniqueValues.py

# update existing and create new unit tests for a test directory
python scripts/bulk_unit_tests_updates.py validmind/tests/data_validation/

# update existing tests only
python scripts/bulk_unit_tests_updates.py validmind/tests/data_validation/ --update-only
```
"""

import os
import subprocess

import click
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

UNIT_TESTS_DIR = os.path.abspath("tests/unit_tests/")
VM_TESTS_DIR = os.path.abspath("validmind/tests/")

OPENAI_MODEL = "gpt-4o"

CREATE_UNIT_TEST_SYSTEM_PROMPT = """
You are an expert software engineer with a strong background in data science and machine learning.
Your task is to create unit tests for a given "ValidMind" test.
ValidMind is a Python library for testing and validating machine learning and other models and datasets.
It provides a test harness alongside a huge library of "tests" that can be used to check and validate many different types of models and datasets.
These tests need their own unit tests to ensure they are working as expected.
You will be given the source code of the "ValidMind" test and your job is to create a unit test for it.
Do not include anything other than the code for the unit test in your response.
Only include the code directly, do not include any backticks or other formatting.
This code will be directly written to a Python file, so make sure it is valid Python code.
Where possible, cache the test result in the setUp method so that it is not run for every test (unless the specific test is using different inputs/parameters).
"""

UPDATE_UNIT_TEST_SYSTEM_PROMPT = """
You are an expert software engineer with a strong background in data science and machine learning.
Your task is to update an existing unit test for a given "ValidMind" test.
ValidMind is a Python library for testing and validating machine learning and other models and datasets.
It provides a test harness alongside a huge library of "tests" that can be used to check and validate many different types of models and datasets.
These tests need their own unit tests to ensure they are working as expected.
You will be given the source code of the "ValidMind" test and the existing unit test for it.
Your job is to update the existing unit test code to work with any updates to the test.
Do not include anything other than the code for the unit test in your response.
Only include the code directly, do not include any backticks or other formatting.
This code will be directly written to a Python file, so make sure it is valid Python code.
If you don't think the existing unit test has any issues, just return the existing unit test code.
The most likely reason for updating the unit test is that something new has been added to the test's return value (e.g. a new table, figure, raw data, etc.)

Note:
- for raw data, you should only check that the raw data is an instance of `vm.RawData` (or `RawData` if you do `from validmind import RawData`)... do not check the contents for now
- only change existing checks if you think they are going to fail or are incorrect

If a unit test doesn't need changes, simply return the exact string "NO CHANGE"!
"""

# SIMPLE_EXAMPLE_TEST_CODE = """# /Users/me/Code/validmind-library/validmind/tests/model_validation/SimpleAccuracy.py

# # Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# # See the LICENSE file in the root of this repository for details.
# # SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

# from sklearn.metrics import accuracy_score

# from validmind.tests import tags, tasks
# from validmind.vm_models import VMDataset, VMModel


# @tags("model_validation")
# @tasks("classification", "regression")
# def SimpleAccuracy(model: VMModel, dataset: VMDataset):
#     y_pred = dataset.y_pred(model)
#     y_true = dataset.y.astype(y_pred.dtype)
#     return accuracy_score(y_true, y_pred)
# """

# SIMPLE_EXAMPLE_UNIT_TEST_CODE = """# /Users/me/Code/validmind-library/tests/unit_tests/model_validation/sklearn/test_SimpleAccuracy.py

# import unittest
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import make_classification
# from validmind.vm_models import VMDataset, VMModel
# from validmind.tests.model_validation.sklearn.SimpleAccuracy import SimpleAccuracy


# class TestSimpleAccuracy(unittest.TestCase):
#     def setUp(self):
#         # Create a synthetic classification dataset
#         X, y = make_classification(
#             n_samples=1000, n_features=10, n_classes=2, random_state=0
#         )

#         # Convert to DataFrame
#         self.df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
#         self.df['target'] = y

#         # Train a simple Logistic Regression model
#         self.model = LogisticRegression()
#         self.model.fit(self.df.drop(columns=["target"]), self.df["target"])

#         # Initialize ValidMind dataset and model
#         self.vm_dataset = VMDataset(input_id="classification_dataset", dataset=self.df, target_column="target", __log=False)
#         self.vm_model = VMModel(input_id="logistic_regression", model=self.model, __log=False)

#         self.result = SimpleAccuracy([self.vm_dataset], self.vm_model)

#     def test_simple_accuracy(self):
#         # Check the types of returned objects
#         self.assertIsInstance(self.result, float)Z
# """


client = OpenAI()


def create_unit_test(vm_test_path, unit_test_path):
    click.echo(f"  Creating new unit test since none exists...")

    # grab a unit test from the same directory
    unit_test_dir = os.path.dirname(unit_test_path)
    unit_test_files = [
        f
        for f in os.listdir(unit_test_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]

    if len(unit_test_files) == 0:
        raise ValueError(
            f"No unit tests exist for the directory {unit_test_dir}."
            " Please create one so we can use it as an example to pass to the LLM"
        )

    eg_unit_test_path = os.path.join(unit_test_dir, unit_test_files[0])

    with open(eg_unit_test_path, "r") as f:
        eg_unit_test_code = f.read()
    eg_unit_test_code = f"# {eg_unit_test_path}\n\n{eg_unit_test_code}"

    # get the associated test file for the example unit test
    eg_vm_test_path = eg_unit_test_path.replace(UNIT_TESTS_DIR, VM_TESTS_DIR).replace(
        "test_", ""
    )

    with open(eg_vm_test_path, "r") as f:
        eg_vm_test_code = f.read()
    eg_vm_test_code = f"# {eg_vm_test_path}\n\n{eg_vm_test_code}"

    # get the vm test file code
    with open(vm_test_path, "r") as f:
        vm_test_code = f.read()
    vm_test_code = f"# {vm_test_path}\n\n{vm_test_code}"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CREATE_UNIT_TEST_SYSTEM_PROMPT},
            {"role": "user", "content": eg_vm_test_code},
            {"role": "assistant", "content": eg_unit_test_code},
            {"role": "user", "content": vm_test_code},
        ],
    )

    unit_test_code = response.choices[0].message.content
    unit_test_code = unit_test_code.replace(f"# {unit_test_path}\n\n", "")
    with open(unit_test_path, "w") as f:
        f.write(unit_test_code)


def update_unit_test(vm_test_path, unit_test_path):
    click.echo(f"  Updating existing unit test...")

    with open(unit_test_path, "r") as f:
        unit_test_code = f.read()

    with open(vm_test_path, "r") as f:
        vm_test_code = f.read()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": UPDATE_UNIT_TEST_SYSTEM_PROMPT},
            {"role": "user", "content": f"# {vm_test_path}\n\n{vm_test_code}"},
            {"role": "user", "content": f"# {unit_test_path}\n\n{unit_test_code}"},
        ],
    )

    new_unit_test_code = response.choices[0].message.content

    if "NO CHANGE" in new_unit_test_code:
        click.echo("No changes needed")
        return

    new_unit_test_code = new_unit_test_code.replace(f"# {unit_test_path}\n\n", "")
    with open(unit_test_path, "w") as f:
        f.write(new_unit_test_code)


def add_or_update_unit_test(vm_test_path, unit_test_path):
    click.echo(f"> {unit_test_path}")

    # check if the unit test file exists
    if not os.path.exists(unit_test_path):
        return create_unit_test(vm_test_path, unit_test_path)

    return update_unit_test(vm_test_path, unit_test_path)


def _is_test_file(path):
    return path.endswith(".py") and path.split("/")[-1][0].isupper()


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--update-only", is_flag=True, help="Only update existing unit tests")
def main(path, update_only):
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
                    tests_to_process.append(os.path.abspath(os.path.join(root, file)))

    # create a tuple of the test path and the associated unit test path
    tests_to_process = [
        (
            test,
            test.replace(VM_TESTS_DIR, UNIT_TESTS_DIR).replace(
                os.path.basename(test), "test_" + os.path.basename(test)
            ),
        )
        for test in tests_to_process
    ]

    if update_only:
        # remove any tests that don't have a unit test
        tests_to_process = [
            (vm_test_path, unit_test_path)
            for vm_test_path, unit_test_path in tests_to_process
            if os.path.exists(unit_test_path)
        ]

    for vm_test_path, unit_test_path in tests_to_process:
        add_or_update_unit_test(vm_test_path, unit_test_path)

    # run black on the tests directory
    subprocess.run(["poetry", "run", "black", UNIT_TESTS_DIR])


if __name__ == "__main__":
    main()
