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
import warnings
from datetime import date, datetime, time
from platform import python_version
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import matplotlib.pylab as pylab
import mistune
import nest_asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from IPython.core import getipython
from IPython.display import HTML
from IPython.display import display as ipy_display
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from numpy import ndarray
from packaging.version import Version
from sklearn.exceptions import UndefinedMetricWarning
from tabulate import tabulate

from .html_templates.content_blocks import math_jax_snippet, python_syntax_highlighting
from .logging import get_logger

DEFAULT_BIG_NUMBER_DECIMALS = 2
DEFAULT_SMALL_NUMBER_DECIMALS = 4

# Suppress some common warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*valid feature names.*"
)

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

T = TypeVar("T")


def parse_version(version: str) -> tuple[int, ...]:
    """
    Parse a semver version string into a tuple of major, minor, patch integers.

    Args:
        version (str): The semantic version string to parse.

    Returns:
        tuple[int, ...]: A tuple of major, minor, patch integers.
    """
    v = Version(version)

    return (v.major, v.minor, v.micro)


def is_notebook() -> bool:
    """
    Checks if the code is running in a Jupyter notebook or IPython shell.

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_handlers = {
            self.is_datetime: lambda obj: obj.isoformat(),
            self.is_pandas_interval: lambda obj: f"[{obj.left}, {obj.right}]",
            self.is_numpy_integer: lambda obj: int(obj),
            self.is_numpy_floating: lambda obj: float(obj),
            self.is_numpy_ndarray: lambda obj: obj.tolist(),
            self.is_numpy_bool: lambda obj: bool(obj),
            self.is_pandas_timestamp: lambda obj: str(obj),
            self.is_numpy_datetime64: lambda obj: str(obj),
            self.is_set: lambda obj: list(obj),
            self.is_quantlib_date: lambda obj: obj.ISO(),
            self.is_generic_object: self.handle_generic_object,
        }

    def default(self, obj):
        for type_check, handler in self.type_handlers.items():
            if type_check(obj):
                return handler(obj)
        return super().default(obj)

    def is_datetime(self, obj):
        return isinstance(obj, (datetime, date, time))

    def is_pandas_interval(self, obj):
        return isinstance(obj, pd.Interval)

    def is_numpy_integer(self, obj):
        return isinstance(obj, np.integer)

    def is_numpy_floating(self, obj):
        return isinstance(obj, np.floating)

    def is_numpy_ndarray(self, obj):
        return isinstance(obj, np.ndarray)

    def is_numpy_bool(self, obj):
        return isinstance(obj, np.bool_)

    def is_pandas_timestamp(self, obj):
        return isinstance(obj, pd.Timestamp)

    def is_numpy_datetime64(self, obj):
        return isinstance(obj, np.datetime64)

    def is_set(self, obj):
        return isinstance(obj, set)

    def is_quantlib_date(self, obj):
        return "QuantLib.Date" in str(type(obj))

    def is_generic_object(self, obj):
        return isinstance(obj, object)

    def handle_generic_object(self, obj):
        try:
            if hasattr(obj, "__str__"):
                return obj.__str__()
            return obj.__class__.__name__
        except Exception:
            return str(type(obj).__name__)

    def encode(self, obj):
        obj = nan_to_none(obj)
        return super().encode(obj)

    def iterencode(self, obj, _one_shot: bool = ...):
        obj = nan_to_none(obj)
        return super().iterencode(obj, _one_shot)


class HumanReadableEncoder(NumpyEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # truncate ndarrays to 10 items
        self.type_handlers[self.is_numpy_ndarray] = lambda obj: (
            obj.tolist()[:5] + ["..."] + obj.tolist()[-5:]
            if len(obj) > 10
            else obj.tolist()
        )

    def default(self, obj):
        if self.is_dataframe(obj):
            return {
                "type": str(type(obj)),
                "preview": obj.head(5).to_dict(orient="list"),
                "shape": f"{obj.shape[0]} rows x {obj.shape[1]} columns",
            }
        return super().default(obj)

    def is_dataframe(self, obj):
        return isinstance(obj, pd.DataFrame)


def get_full_typename(o: Any) -> Any:
    """We determine types based on type names so we don't have to import."""
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


def format_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
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


def format_key_values(key_values: Dict[str, Any]) -> Dict[str, Any]:
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
    TODO: generalize this to work with metrics and test results.

    Summarize the results of the data quality test suite.
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
    """Format a pandas DataFrame for display purposes."""
    df = df.style.set_properties(**{"text-align": "left"}).hide(axis="index")
    return df.set_table_styles([dict(selector="th", props=[("text-align", "left")])])


def run_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> T:
    """Helper function to run functions asynchronously.

    This takes care of the complexity of running the logging functions asynchronously. It will
    detect the type of environment we are running in (IPython notebook or not) and run the
    function accordingly.

    Args:
        func: The function to run asynchronously.
        *args: The arguments to pass to the function.
        name: Optional name for the task.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The result of the function.
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


def run_async_check(
    func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
) -> Optional[asyncio.Task[T]]:
    """Helper function to run functions asynchronously if the task doesn't already exist.

    Args:
        func: The function to run asynchronously.
        *args: The arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        Optional[asyncio.Task[T]]: The task if created or found, None otherwise.
    """
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


def fuzzy_match(string: str, search_string: str, threshold: float = 0.7) -> bool:
    """Check if a string matches another string using fuzzy matching.

    Args:
        string (str): The string to check.
        search_string (str): The string to search for.
        threshold (float): The similarity threshold to use (Default: 0.7).

    Returns:
        bool: True if the string matches the search string, False otherwise.
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
    """Attempts to extract all model info from a model object instance."""
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
    """Attempts to extract all dataset info from a dataset object instance."""
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
    """Preview test configuration in a collapsible HTML section.

    Args:
        config (dict): Test configuration dictionary.
    """

    try:
        formatted_json = json.dumps(serialize(config), indent=4)
    except TypeError as e:
        logger.error(f"JSON serialization failed: {e}")
        return

    collapsible_html = f"""
    <script>
    function toggleOutput() {{
        var content = document.getElementById("collapsibleContent");
        content.style.display = content.style.display === "none" ? "block" : "none";
    }}
    </script>
    <button onclick="toggleOutput()">Preview Config</button>
    <div id="collapsibleContent" style="display:none;"><pre>{formatted_json}</pre></div>
    """

    ipy_display(HTML(collapsible_html))


def display(widget_or_html, syntax_highlighting=True, mathjax=True):
    """Display HTML content with extra goodies (syntax highlighting, MathJax, etc.)."""
    if hasattr(widget_or_html, "to_html"):
        html_content = widget_or_html.to_html()
        ipy_display(HTML(html_content))
        syntax_highlighting = 'class="language-' in html_content
        mathjax = "math/tex" in html_content
    elif isinstance(widget_or_html, str):
        ipy_display(HTML(widget_or_html))
        syntax_highlighting = 'class="language-' in widget_or_html
        mathjax = "math/tex" in widget_or_html
    else:
        # Fallback: convert to string representation
        ipy_display(HTML(str(widget_or_html)))

    if syntax_highlighting:
        ipy_display(HTML(python_syntax_highlighting))

    if mathjax:
        ipy_display(HTML(math_jax_snippet))


def md_to_html(md: str, mathml=False) -> str:
    """Converts Markdown to HTML using mistune with plugins."""
    # use mistune with math plugin to convert to html
    html = mistune.create_markdown(
        plugins=["math", "table", "strikethrough", "footnotes"]
    )(md)

    if not mathml:
        return html

    # convert the latex to mathjax
    math_block_pattern = re.compile(r'<div class="math">\$\$([\s\S]*?)\$\$</div>')
    html = math_block_pattern.sub(
        lambda match: '<script type="math/tex; mode=display">{}</script>'.format(
            match.group(1)
        ),
        html,
    )

    inline_math_pattern = re.compile(r'<span class="math">\\\((.*?)\\\)</span>')
    html = inline_math_pattern.sub(
        lambda match: '<script type="math/tex">{}</script>'.format(match.group(1)),
        html,
    )

    return html


def is_html(text: str) -> bool:
    """Check if a string is HTML.

    Uses more robust heuristics to determine if a string contains HTML content.

    Args:
        text (str): The string to check

    Returns:
        bool: True if the string likely contains HTML, False otherwise
    """
    # Strip whitespace first
    text = text.strip()

    # Basic check: Must at least start with < and end with >
    if not (text.startswith("<") and text.endswith(">")):
        return False

    # Look for common HTML tags
    common_html_patterns = [
        r"<html.*?>",  # HTML tag
        r"<body.*?>",  # Body tag
        r"<div.*?>",  # Div tag
        r"<p>.*?</p>",  # Paragraph with content
        r"<h[1-6]>.*?</h[1-6]>",  # Headers
        r"<script.*?>",  # Script tags
        r"<style.*?>",  # Style tags
        r"<a href=.*?>",  # Links
        r"<img.*?>",  # Images
        r"<table.*?>",  # Tables
        r"<!DOCTYPE html>",  # DOCTYPE declaration
    ]

    for pattern in common_html_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return True

    # If we have at least 2 matching tags, it's likely HTML
    # This helps detect custom elements or patterns not in our list
    tags = re.findall(r"</?[a-zA-Z][a-zA-Z0-9]*.*?>", text)
    if len(tags) >= 2:
        return True

    # Try parsing with BeautifulSoup as a last resort
    try:
        soup = BeautifulSoup(text, "html.parser")
        # If we find any tags that weren't in the original text, BeautifulSoup
        # likely tried to fix broken HTML, meaning it's not valid HTML
        return len(soup.find_all()) > 0

    except Exception as e:
        logger.error(f"Error checking if text is HTML: {e}")
        return False

    return False


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


def serialize(obj):
    """Convert objects to JSON-serializable format with readable descriptions."""
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return ""  # Simple empty string for non-serializable objects
    return obj


def is_text_column(series, threshold=0.05) -> bool:
    """
    Determines if a series is likely to contain text data using heuristics.

    Args:
        series (pd.Series): The pandas Series to analyze
        threshold (float): The minimum threshold to classify a pattern match as significant

    Returns:
        bool: True if the series likely contains text data, False otherwise
    """
    # Filter to non-null string values and sample if needed
    string_series = series.dropna().astype(str)
    if len(string_series) == 0:
        return False
    if len(string_series) > 1000:
        string_series = string_series.sample(1000, random_state=42)

    # Calculate basic metrics
    total_values = len(string_series)
    unique_ratio = len(string_series.unique()) / total_values if total_values > 0 else 0
    avg_length = string_series.str.len().mean()
    avg_words = string_series.str.split(r"\s+").str.len().mean()

    # Check for special text patterns
    patterns = {
        "url": r"https?://\S+|www\.\S+",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "filepath": r'(?:[a-zA-Z]:|[\\/])(?:[\\/][^\\/:*?"<>|]+)+',
    }

    # Check if any special patterns exceed threshold
    for pattern in patterns.values():
        if string_series.str.contains(pattern, regex=True, na=False).mean() > threshold:
            return True

    # Calculate proportion of alphabetic characters
    total_chars = string_series.str.len().sum()
    if total_chars > 0:
        alpha_ratio = string_series.str.count(r"[a-zA-Z]").sum() / total_chars
    else:
        alpha_ratio = 0

    # Check for free-form text indicators
    text_indicators = [
        unique_ratio > 0.8 and avg_length > 20,  # High uniqueness and long strings
        unique_ratio > 0.4
        and avg_length > 15
        and string_series.str.contains(r"[.,;:!?]", regex=True, na=False).mean()
        > 0.3,  # Moderate uniqueness with punctuation
        string_series.str.contains(
            r"\b\w+\b\s+\b\w+\b\s+\b\w+\b\s+\b\w+\b", regex=True, na=False
        ).mean()
        > 0.3,  # Contains long phrases
        avg_words > 5 and alpha_ratio > 0.6,  # Many words with mostly letters
        unique_ratio > 0.95 and avg_length > 10,  # Very high uniqueness
    ]

    return any(text_indicators)


def _get_numeric_type_detail(column, dtype, series):
    """Helper function to determine numeric type details."""
    if pd.api.types.is_integer_dtype(dtype):
        return {"type": "Numeric", "subtype": "Integer"}
    elif pd.api.types.is_float_dtype(dtype):
        return {"type": "Numeric", "subtype": "Float"}
    else:
        return {"type": "Numeric", "subtype": "Other"}


def _get_text_type_detail(series):
    """Helper function to determine text/categorical type details."""
    string_series = series.dropna().astype(str)

    if len(string_series) == 0:
        return {"type": "Categorical"}

    # Check for common patterns
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    filepath_pattern = r'(?:[a-zA-Z]:|[\\/])(?:[\\/][^\\/:*?"<>|]+)+'

    url_ratio = string_series.str.contains(url_pattern, regex=True, na=False).mean()
    email_ratio = string_series.str.contains(email_pattern, regex=True, na=False).mean()
    filepath_ratio = string_series.str.contains(
        filepath_pattern, regex=True, na=False
    ).mean()

    # Check if general text using enhanced function
    if url_ratio > 0.7:
        return {"type": "Text", "subtype": "URL"}
    elif email_ratio > 0.7:
        return {"type": "Text", "subtype": "Email"}
    elif filepath_ratio > 0.7:
        return {"type": "Text", "subtype": "Path"}
    elif is_text_column(series):
        return {"type": "Text", "subtype": "FreeText"}

    # Must be categorical
    n_unique = series.nunique()
    if n_unique == 2:
        return {"type": "Categorical", "subtype": "Binary"}
    else:
        return {"type": "Categorical", "subtype": "Nominal"}


def get_column_type_detail(df, column) -> dict:
    """
    Get detailed column type information beyond basic type detection.
    Similar to ydata-profiling's type system.

    Args:
        df (pd.DataFrame): DataFrame containing the column
        column (str): Column name to analyze

    Returns:
        dict: Detailed type information including primary type and subtype
    """
    series = df[column]
    dtype = series.dtype

    # Initialize result with id and basic type
    result = {"id": column, "type": "Unknown"}

    # Determine type details based on dtype
    type_detail = None

    if pd.api.types.is_numeric_dtype(dtype):
        type_detail = _get_numeric_type_detail(column, dtype, series)
    elif pd.api.types.is_bool_dtype(dtype):
        type_detail = {"type": "Boolean"}
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        type_detail = {"type": "Datetime"}
    elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(
        dtype
    ):
        type_detail = _get_text_type_detail(series)

    # Update result with type details
    if type_detail:
        result.update(type_detail)

    return result


def infer_datatypes(df, detailed=False) -> list:
    """
    Infer data types for columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze
        detailed (bool): Whether to return detailed type information including subtypes

    Returns:
        list: Column type mappings
    """
    if detailed:
        return [get_column_type_detail(df, column) for column in df.columns]

    column_type_mappings = {}
    # Use pandas to infer data types
    for column in df.columns:
        # Check if all values are None
        if df[column].isna().all():
            column_type_mappings[column] = {"id": column, "type": "Null"}
            continue

        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            column_type_mappings[column] = {"id": column, "type": "Numeric"}
        elif pd.api.types.is_bool_dtype(dtype):
            column_type_mappings[column] = {"id": column, "type": "Boolean"}
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_type_mappings[column] = {"id": column, "type": "Datetime"}
        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(
            dtype
        ):
            # Check if this is more likely to be text than categorical
            if is_text_column(df[column]):
                column_type_mappings[column] = {"id": column, "type": "Text"}
            else:
                column_type_mappings[column] = {"id": column, "type": "Categorical"}
        else:
            column_type_mappings[column] = {"id": column, "type": "Unsupported"}

    return list(column_type_mappings.values())
