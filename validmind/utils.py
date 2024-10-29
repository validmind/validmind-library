# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import asyncio
import difflib
import inspect
import json
import math
import re
import sys
from platform import python_version
from typing import Any

import matplotlib.pylab as pylab
import mistune
import nest_asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core import getipython
from IPython.display import HTML
from IPython.display import display as ipy_display
from latex2mathml.converter import convert
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from numpy import ndarray
from tabulate import tabulate

from .html_templates.content_blocks import math_jax_snippet, python_syntax_highlighting
from .logging import get_logger

DEFAULT_BIG_NUMBER_DECIMALS = 2
DEFAULT_SMALL_NUMBER_DECIMALS = 4


# SETUP SOME DEFAULTS FOR PLOTS #
# Silence this warning: *c* argument looks like a single numeric RGB or
# RGBA sequence, which should be avoided
matplotlib_axes_logger.setLevel("ERROR")

sns.set(rc={"figure.figsize": (20, 10)})

params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)
#################################

logger = get_logger(__name__)


def is_notebook() -> bool:
    """
    Checks if the code is running in a Jupyter notebook or IPython shell

    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        if getipython.get_ipython() is not None:
            return True
    except NameError:
        return False  # Probably standard Python interpreter

    return False


# hacky way to make async code run "synchronously" in colab
__loop: asyncio.AbstractEventLoop = None
try:
    from google.colab._shell import Shell  # type: ignore

    if isinstance(getipython.get_ipython(), Shell):
        __loop = asyncio.new_event_loop()
        nest_asyncio.apply(__loop)
except ModuleNotFoundError:
    if is_notebook():
        __loop = asyncio.new_event_loop()
        nest_asyncio.apply(__loop)


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Interval):
            return f"[{obj.left}, {obj.right}]"
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

    def encode(self, obj):
        obj = nan_to_none(obj)
        return super().encode(obj)

    def iterencode(self, obj, _one_shot: bool = ...):
        obj = nan_to_none(obj)
        return super().iterencode(obj, _one_shot)


def get_full_typename(o: Any) -> Any:
    """We determine types based on type names so we don't have to import
    (and therefore depend on) PyTorch, TensorFlow, etc.
    """
    instance_name = o.__class__.__module__ + "." + o.__class__.__name__
    if instance_name in ["builtins.module", "__builtin__.module"]:
        return o.__name__
    else:
        return instance_name


def is_matplotlib_typename(typename: str) -> bool:
    return typename.startswith("matplotlib.")


def is_plotly_typename(typename: str) -> bool:
    return typename.startswith("plotly.")


def precision_and_scale(x):
    """
    https://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python

    Returns a (precision, scale) tuple for a given number.
    """
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return (magnitude + scale, scale)


def format_records(df):
    """
    Round the values on each dataframe's column to a given number of decimal places.
    The returned value is converted to a dict in "records" with Pandas's to_dict() function.

    We do this for display purposes before sending data to ValidMind. Rules:

    - Check if we are rendering "big" numbers greater than 10 or just numbers between 0 and 1
    - If the column's smallest number has more decimals 6, use that number's precision
      so we can avoid rendering a 0 instead
    - If the column's smallest number has less decimals than 6, use 6 decimal places
    """
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        not_zero = df[col][df[col] != 0]
        min_number = not_zero.min()
        if math.isnan(min_number) or math.isinf(min_number):
            df[col] = df[col].round(DEFAULT_SMALL_NUMBER_DECIMALS)
            continue

        _, min_scale = precision_and_scale(min_number)

        if min_number >= 10:
            df[col] = df[col].round(DEFAULT_BIG_NUMBER_DECIMALS)
        elif min_scale > DEFAULT_SMALL_NUMBER_DECIMALS:
            df[col] = df[col].round(DEFAULT_SMALL_NUMBER_DECIMALS)
        else:
            df[col] = df[col].round(min_scale)

    return df.to_dict("records")


def format_key_values(key_values):
    """
    Round the values on each dict's value to a given number of decimal places.

    We do this for display purposes before sending data to ValidMind. Rules:

    - Assume the dict is in this form: {key1: value1, key2: value2, ...}
    - Check if we are rendering "big" numbers greater than 10 or just numbers between 0 and 1
    - If the column's smallest number has more decimals 6, use that number's precision
      so we can avoid rendering a 0 instead
    - If the column's smallest number has less decimals than 6, use 6 decimal places
    """
    min_number = min([v for v in key_values.values() if v != 0])
    _, min_scale = precision_and_scale(min_number)

    for key, value in key_values.items():
        # Some key values could be a single item ndarray, assert this
        if isinstance(value, ndarray):
            assert len(value) == 1, "Expected a single item ndarray"
            value = value[0]

        if min_number >= 10:
            key_values[key] = round(value, DEFAULT_BIG_NUMBER_DECIMALS)
        elif min_scale > DEFAULT_SMALL_NUMBER_DECIMALS:
            key_values[key] = round(value, DEFAULT_SMALL_NUMBER_DECIMALS)
        else:
            key_values[key] = round(value, min_scale)

    return key_values


def summarize_data_quality_results(results):
    """
    TODO: generalize this to work with metrics and test results

    Summarize the results of the data quality test suite
    """
    test_results = []
    for result in results:
        num_passed = len([r for r in result.results if r.passed])
        num_failed = len([r for r in result.results if not r.passed])

        percent_passed = (
            1 if len(result.results) == 0 else num_passed / len(result.results)
        )
        test_results.append(
            [
                result.test_name,
                result.passed,
                num_passed,
                num_failed,
                percent_passed * 100,
            ]
        )

    return tabulate(
        test_results,
        headers=["Test", "Passed", "# Passed", "# Errors", "% Passed"],
        numalign="right",
    )


def format_number(number):
    """
    Format a number for display purposes. If the number is a float, round it
    to 4 decimal places.
    """
    if isinstance(number, float):
        return round(number, 4)
    else:
        return number


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format a pandas DataFrame for display purposes"""
    df = df.style.set_properties(**{"text-align": "left"}).hide(axis="index")
    return df.set_table_styles([dict(selector="th", props=[("text-align", "left")])])


def run_async(func, *args, name=None, **kwargs):
    """Helper function to run functions asynchronously

    This takes care of the complexity of running the logging functions asynchronously. It will
    detect the type of environment we are running in (ipython notebook or not) and run the
    function accordingly.

    Args:
        func (function): The function to run asynchronously
        *args: The arguments to pass to the function
        **kwargs: The keyword arguments to pass to the function

    Returns:
        The result of the function
    """
    try:
        if asyncio.get_event_loop().is_running() and is_notebook():
            if __loop:
                future = __loop.create_task(func(*args, **kwargs), name=name)
                # wait for the future result
                return __loop.run_until_complete(future)

            return asyncio.get_event_loop().create_task(
                func(*args, **kwargs), name=name
            )
    except RuntimeError:
        pass

    return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))


def run_async_check(func, *args, **kwargs):
    """Helper function to run functions asynchronously if the task doesn't already exist"""
    if __loop:
        return  # we don't need this if we are using our own loop

    try:
        name = func.__name__

        for task in asyncio.all_tasks():
            if task.get_name() == name:
                return task

        return run_async(func, name=name, *args, **kwargs)  # noqa B026

    except RuntimeError:
        pass


def fuzzy_match(string: str, search_string: str, threshold=0.7):
    """Check if a string matches another string using fuzzy matching

    Args:
        string (str): The string to check
        search_string (str): The string to search for
        threshold (float): The similarity threshold to use (Default: 0.7)

    Returns:
        True if the string matches the search string, False otherwise
    """
    score = difflib.SequenceMatcher(None, string, search_string).ratio()

    return score >= threshold


def test_id_to_name(test_id: str) -> str:
    """Convert a test ID to a human-readable name.

    Args:
        test_id (str): The test identifier, typically in CamelCase or snake_case.

    Returns:
        str: A human-readable name derived from the test ID.
    """
    last_part = test_id.split(".")[-1]
    words = []

    # Split on underscores and apply regex to each part to handle CamelCase and acronyms
    for part in last_part.split("_"):
        # Regex pattern to match uppercase acronyms, mixed-case words, or alphanumeric combinations
        words.extend(
            re.findall(r"[A-Z]+(?:_[A-Z]+)*(?=_|$|[A-Z][a-z])|[A-Z]?[a-z0-9]+", part)
        )

    # Join the words with spaces, capitalize non-acronym words
    return " ".join(word.capitalize() if not word.isupper() else word for word in words)


def get_model_info(model):
    """Attempts to extract all model info from a model object instance"""
    architecture = model.name
    framework = model.library
    framework_version = model.library_version
    language = model.language

    if language is None:
        language = f"Python {python_version()}"

    if framework_version == "N/A" or framework_version is None:
        try:
            framework_version = sys.modules[framework].__version__
        except (KeyError, AttributeError):
            framework_version = "N/A"

    return {
        "architecture": architecture,
        "framework": framework,
        "framework_version": framework_version,
        "language": language,
    }


def get_dataset_info(dataset):
    """Attempts to extract all dataset info from a dataset object instance"""
    num_rows, num_cols = dataset.df.shape
    schema = dataset.df.dtypes.apply(lambda x: x.name).to_dict()
    description = (
        dataset.df.describe(include="all").reset_index().to_dict(orient="records")
    )

    return {
        "num_rows": num_rows,
        "num_columns": num_cols,
        "schema": schema,
        "description": description,
    }


def preview_test_config(config):
    formatted_json = json.dumps(config, indent=4)

    # JavaScript + HTML for the collapsible section
    collapsible_html = f"""
    <script>
    function toggleOutput() {{
        var content = document.getElementById("collapsibleContent");
        if (content.style.display === "none") {{
            content.style.display = "block";
        }} else {{
            content.style.display = "none";
        }}
    }}
    </script>
    <button onclick="toggleOutput()">Preview Config</button>
    <div id="collapsibleContent" style="display:none;"><pre>{formatted_json}</pre></div>
    """

    ipy_display(HTML(collapsible_html))


def display(widget_or_html, syntax_highlighting=True, mathjax=True):
    """Display widgets with extra goodies (syntax highlighting, MathJax, etc.)"""
    if isinstance(widget_or_html, str):
        ipy_display(HTML(widget_or_html))
        # if html we can auto-detect if we actually need syntax highlighting or MathJax
        syntax_highlighting = 'class="language-' in widget_or_html
        mathjax = "$$" in widget_or_html
    else:
        ipy_display(widget_or_html)

    if syntax_highlighting:
        ipy_display(HTML(python_syntax_highlighting))

    if mathjax:
        ipy_display(HTML(math_jax_snippet))


def md_to_html(md: str, mathml=False) -> str:
    """Converts Markdown to HTML using mistune with plugins"""
    # use mistune with math plugin to convert to html
    html = mistune.create_markdown(
        plugins=["math", "table", "strikethrough", "footnotes"]
    )(md)

    if not mathml:
        # return the html as is (with latex that will be rendered by MathJax)
        return html

    # convert the latex to MathML which CKeditor can render
    math_block_pattern = re.compile(r'<div class="math">\$\$([\s\S]*?)\$\$</div>')
    html = math_block_pattern.sub(
        lambda match: "<p>{}</p>".format(convert(match.group(1), display="block")), html
    )

    inline_math_pattern = re.compile(r'<span class="math">\\\((.*?)\\\)</span>')
    html = inline_math_pattern.sub(
        lambda match: "<span>{}</span>".format(
            convert(match.group(1), display="inline")
        ),
        html,
    )

    return html


def inspect_obj(obj):
    # Filtering only attributes
    print(len("Attributes:") * "-")
    print("Attributes:")
    print(len("Attributes:") * "-")

    # Get only attributes (not methods)
    attributes = [
        attr
        for attr in dir(obj)
        if not callable(getattr(obj, attr)) and not attr.startswith("__")
    ]
    for attr in attributes:
        print(f"{attr}")

    # Filtering only methods using inspect and displaying their parameters
    print("\nMethods with Parameters:")

    # Get only methods (functions) using inspect.ismethod
    methods = inspect.getmembers(obj, predicate=inspect.ismethod)
    print("Methods:")
    for name, method in methods:
        # Get the signature of the method
        sig = inspect.signature(method)
        print(len(f"{name}") * "-")
        print(f"{name}")
        print(len(f"{name}") * "-")
        print("Parameters:")
        # Loop through the parameters and print detailed information
        for param_name, param in sig.parameters.items():
            print(f"{param_name} - ({param.default})")
