"""Script to run notebooks for integration testing the library.

Usage:
    python scripts/run_e2e_notebooks.py

Note: This script is meant to be run from the root of the repo

Notebooks Tested:
 - notebooks/code_samples/quickstart_customer_churn_full_suite.ipynb
 - notebooks/code_samples/time_series/quickstart_time_series_full_suite.ipynb
 - notebooks/code_samples/regression/quickstart_regression_full_suite.ipynb
 - notebooks/code_samples/custom_tests/external_test_providers.ipynb
 - notebooks/code_samples/custom_tests/implement_custom_tests.ipynb

To add more notebooks to the list, simply add the path to the `NOTEBOOKS_TO_RUN` list.
This will use the default project id for the notebook. If you want to use a different
project/template, you can specify it as a dictionary with the keys `path` and `project`
where `path` is the path to the notebook and `project` is the project id.

Note: This script requires the following environment variables to be set:
    - NOTEBOOK_RUNNER_API_KEY (api key for the project)
    - NOTEBOOK_RUNNER_API_SECRET (api secret for the project)

This uses the dev environment for now... In the future, we may want to change this???
"""

import os
import re

import click
import dotenv
import nbformat
import papermill as pm

dotenv.load_dotenv()

DEFAULT_MODEL_CUID = os.getenv(
    "NOTEBOOK_RUNNER_DEFAULT_MODEL", "cltnl28x600001omg9wu8wfty"
)  # Demo Account Dev Customer Churn Model

NOTEBOOKS_TO_RUN = [
    "notebooks/code_samples/quickstart_customer_churn_full_suite.ipynb",
    "notebooks/code_samples/time_series/quickstart_time_series_full_suite.ipynb",
    "notebooks/code_samples/regression/quickstart_regression_full_suite.ipynb",
    "notebooks/how_to/run_unit_metrics.ipynb",
    "notebooks/code_samples/custom_tests/integrate_external_test_providers.ipynb",
    "notebooks/code_samples/custom_tests/implement_custom_tests.ipynb",
]

INIT_CELL_CODE = """
import os
os.environ["VALIDMIND_LLM_DESCRIPTIONS_ENABLED"] = "0"
import validmind as vm

vm.init(
  api_host = "{api_host}",
  api_key = "{api_key}",
  api_secret = "{api_secret}",
  model = "{model}"
)


import logging
import sys
import site

logger = logging.getLogger()

# Print the path to the current Python interpreter
logger.info("Python executable path: " + sys.executable)

# Print the path to the site-packages directory
logger.info("Site-packages path: " + str(site.getsitepackages()))
"""


@click.command()
@click.option(
    "--kernel", default="python3", help="Kernel to use when executing notebooks."
)
@click.option(
    "--log-output",
    is_flag=True,
    default=False,
    help="Log the output of the notebook to the console.",
)
@click.option(
    "--progress-bar",
    is_flag=True,
    default=True,
    help="Show the progress bar when executing notebooks.",
)
def main(kernel, log_output=False, progress_bar=True):
    """Run notebooks from the specified directory for end-to-end testing."""
    for notebook_file in NOTEBOOKS_TO_RUN:
        if isinstance(notebook_file, dict):
            notebook_path = os.path.join(os.getcwd(), notebook_file["path"])
            model = notebook_file["model"]
        else:
            notebook_path = os.path.join(os.getcwd(), notebook_file)
            model = DEFAULT_MODEL_CUID

        backup_notebook(notebook_path)

        try:
            update_vm_init_cell(notebook_path, model)
            click.echo(f"\n -------- Executing {notebook_path} ---------- \n")
            run_notebook(
                notebook_path=notebook_path,
                kernel_name=kernel,
                log_output=log_output,
                progress_bar=progress_bar,
            )
            click.echo(f" -------- Finished executing {notebook_path} ---------- \n")
        except Exception as e:
            click.echo(f"Error running {notebook_path}: {e}")
            os.remove(notebook_path.replace(".ipynb", ".out.ipynb"))
            restore_notebook(notebook_path)
            raise e

        restore_notebook(notebook_path)


def run_notebook(notebook_path, kernel_name, log_output=False, progress_bar=True):
    output_path = notebook_path.replace(".ipynb", ".out.ipynb")

    is_gh_actions = os.getenv("GITHUB_ACTIONS") == "true"

    pm.execute_notebook(
        input_path=notebook_path,
        output_path=output_path,
        kernel_name=kernel_name,
        log_output=log_output or is_gh_actions,
        progress_bar=progress_bar or (not is_gh_actions and not log_output),
        cwd=os.path.dirname(notebook_path),
    )

    # comment out the below line to see output notebook for debugging
    os.remove(output_path)


def update_vm_init_cell(notebook_path, model):
    api_host = os.getenv(
        "NOTEBOOK_RUNNER_API_HOST", "https://api.dev.vm.validmind.ai/api/v1/tracking"
    )
    api_key = os.getenv("NOTEBOOK_RUNNER_API_KEY")
    api_secret = os.getenv("NOTEBOOK_RUNNER_API_SECRET")

    init_code = INIT_CELL_CODE.format(
        api_host=api_host,
        api_key=api_key,
        api_secret=api_secret,
        model=model,
    )

    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "vm.init(" in cell["source"]:
            # replace any existing vm.init() calls with the new one
            cell["source"] = re.sub(
                r"vm.init\(.+\)", init_code, cell["source"], flags=re.DOTALL
            )

    with open(notebook_path, "w") as f:
        nbformat.write(nb, f)


def backup_notebook(notebook_path):
    backup_path = f"{notebook_path}.backup"
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.system(f"cp {notebook_path} {backup_path}")


def restore_notebook(notebook_path):
    backup_path = f"{notebook_path}.backup"
    os.rename(backup_path, notebook_path)


if __name__ == "__main__":
    main()
