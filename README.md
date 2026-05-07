# ValidMind Library

<!-- TODO: put back in when workflows are working properly -->
<!-- [![Code Quality](https://github.com/validmind/validmind-library/actions/workflows/python.yaml/badge.svg)](https://github.com/validmind/validmind-library/actions/workflows/python.yaml)
[![Integration Tests](https://github.com/validmind/validmind-library/actions/workflows/integration.yaml/badge.svg)](https://github.com/validmind/validmind-library/actions/workflows/integration.yaml) -->

![ValidMind logo](https://vmai.s3.us-west-1.amazonaws.com/validmind-logo.svg "ValidMind logo")

The ValidMind Library is a suite of developer tools and methods designed to run validation tests and automate the documentation of your models.

Designed to be model agnostic, the ValidMind Library provides all the standard functionality without requiring you to rewrite any functions as long as your model is built in Python.

With a rich array of documentation tools and test suites, from documenting descriptions of your datasets to testing your models for weak spots and overfit areas, the ValidMind Library helps you automate model documentation by feeding the ValidMind Platform with documentation artifacts and test results.

## Contributing to the ValidMind Library

We believe in the power of collaboration and welcome contributions to the ValidMind Library. If you've noticed a bug, have a feature request, or want to contribute a test, please create a pull request or submit an issue and refer to the [contributing guide](README.md#how-to-contribute) below.

- Interested in connecting with fellow AI model risk practitioners? Join our [Community Slack](https://docs.validmind.ai/about/contributing/join-community.html)!

- For more information about ValidMind's open-source tests and Jupyter Notebooks, read the [ValidMind Library docs](https://docs.validmind.ai/developer/validmind-library.html).

## Getting started

### Install from PyPI

To install the ValidMind Library and all optional dependencies, run:

```bash
pip install validmind[all]
```

To just install the core functionality without optional dependencies (some tests and models may not work), run:

```bash
pip install validmind
```

#### Extra dependencies

- **Install with LLM Support**

    ```bash
    pip install validmind[llm]
    ```

- **Install with Hugging Face `transformers` support**

    ```bash
    pip install validmind[transformers]
    ```

- **Install with PyTorch support**

    ```bash
    pip install validmind[pytorch]
    ```

- **Install with R support (requires R to be installed)**

    ```bash
    pip install validmind
    pip install rpy2
    ```

## PII Detection

The ValidMind Library includes optional PII detection capabilities using Microsoft Presidio to automatically detect sensitive data in test results and prevent accidental logging.

**Installation:**

```bash
pip install validmind[pii-detection]
```

**Configure PII detection:**

```bash
# Enable PII detection for test results only
export VALIDMIND_PII_DETECTION=test_results

# Enable PII detection for test descriptions only
export VALIDMIND_PII_DETECTION=test_descriptions

# Enable PII detection for both test results and descriptions
export VALIDMIND_PII_DETECTION=all

# Disable PII detection (default)
export VALIDMIND_PII_DETECTION=disabled
```

## How to contribute

### Install dependencies

- Ensure you have `poetry` installed: <https://python-poetry.org/>

- After cloning this repo, run:

```bash
make install
```

This will install the dependencies and git hooks for the project.

- To run Jupyter notebooks using the source code from the repo, you can use `poetry` to register
a new kernel with Jupyter:

```bash
poetry run python -m ipykernel install --user --name validmind --display-name "ValidMind Library"
```

### Installing LLM validation dependencies

You can install the `transformers`, `torch` and `openai` dependencies using the `llm` extra. This will install the Hugging Face transformers and PyTorch libraries as well as the OpenAI SDK for running the LLM validation examples:

```bash
poetry install --extras llm
```

### Setting up R support

#### 1. Install R

You can download R from <https://cran.r-project.org/>. On macOS, the easiest way is via Homebrew:

```bash
brew install r
```

#### 2. Install Python dependencies

Install `rpy2` so the Python library can interface with R models. On macOS, you may need to build from source to match your R version:

```bash
# Try the standard install first
pip install rpy2

# If you get R library loading errors, rebuild against your installed R:
R_HOME=$(Rscript -e 'cat(R.home())') pip install --no-binary :all: --force-reinstall rpy2
```

#### 3. Install R packages

Open R (type `R` in your terminal) and install the required packages:

```r
install.packages(c("reticulate", "dplyr", "caTools", "knitr", "glue", "plotly", "htmltools", "rmarkdown", "DT", "base64enc"))
```

Then install the ValidMind R package from source:

```r
install.packages("r/validmind", repos = NULL, type = "source")
```

#### 4. Set up VS Code / Cursor for R

No RStudio required. Install the **R extension** (`REditorSupport.r`) in VS Code or Cursor:

1. Open Extensions (`Cmd+Shift+X`) and search for "R"
2. Install the **R** extension by REditorSupport
3. Optionally install the `languageserver` R package for autocomplete: `install.packages("languageserver")`

With the extension installed:
- Open `.Rmd` files and run chunks with `Cmd+Shift+Enter`
- Render full documents with `Cmd+Shift+K`
- Use the R terminal panel for interactive sessions

Alternatively, you can run R notebooks as Jupyter notebooks by installing the R kernel:

```r
install.packages("IRkernel")
IRkernel::installspec()
```

Then create/open `.ipynb` files in VS Code and select the R kernel.

#### 5. Run the quickstart notebooks

Launch R from the repository root (so dataset paths resolve correctly) and run through the notebooks in `notebooks/code_sharing/r/`:

- `quickstart_model_documentation.Rmd` — model documentation workflow
- `quickstart_model_validation.Rmd` — model validation workflow

### Versioning

Make sure you bump the package version before merging a PR with the following command:

```bash
make version tag=patch
```

The value of `tag` corresponds to one of the options provided by Poetry: <https://python-poetry.org/docs/cli/#version>

## Generating summaries for test descriptions

Use `add_test_description.py` to generate a draft descriptions for a test using ChatGPT. This will automatically insert the description into the `class` docstring.

Entire directory:

```bash
poetry run python scripts/add_test_description.py review validmind/tests/example_directory/
```

Single file:

```bash
poetry run python scripts/add_test_description.py review validmind/tests/ongoing_monitoring/FeatureDrift.py
```

## Adding a copyright header

When adding new Python or stand-alone Jupyter Notebook files to the project, you can add the ValidMind copyright header to any files that
are missing it by running:

```bash
make copyright
```

## Known issues

### ValidMind wheel errors

If you run into an error related to the ValidMind wheel, try:

```bash
poetry add wheel
poetry update wheel
poetry install
```
