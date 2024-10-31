# ValidMind Library

<!-- TODO: put back in when workflows are working properly -->
<!-- [![Code Quality](https://github.com/validmind/developer-framework/actions/workflows/python.yaml/badge.svg)](https://github.com/validmind/developer-framework/actions/workflows/python.yaml)
[![Integration Tests](https://github.com/validmind/developer-framework/actions/workflows/integration.yaml/badge.svg)](https://github.com/validmind/developer-framework/actions/workflows/integration.yaml) -->

![ValidMind logo](images/ValidMind-logo-color.svg "ValidMind logo")

ValidMind's Library contains a suite of developer tools and methods designed to run validation tests and automate the documentation of your models. The Library provides a rich suite of documentation tools and test suites, from documenting descriptions of your dataset to testing your models for weak spots and overfit areas.

The Library includes the Python client library, which is designed to be model agnostic. If your model is built in Python, ValidMind's Python library will provide all the standard functionality without requiring your developers to rewrite any functions.

## Contributing to the ValidMind Library

We believe in the power of collaboration and welcome contributions to the ValidMind Library. If you've noticed a bug, have a feature request, or want to contribute a test, please create a pull request or submit an issue and refer to the [contributing guide](README.md#how-to-contribute) below.

- Interested in connecting with fellow AI model risk practitioners? Join our [Community Slack](https://docs.validmind.ai/about/contributing/join-community.html)!

- For more information about ValidMind's open source tests and Jupyter notebooks, read the [Library docs](https://docs.validmind.ai/developer/get-started-developer-framework.html).

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
    pip install validmind[r-support]
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
poetry run python -m ipykernel install --user --name dev-framework --display-name "Library"
```

### Installing LLM validation dependencies

You can install the `transformers`, `torch` and `openai` dependencies using the `llm` extra. This will install the Hugging Face transformers and PyTorch libraries as well as the OpenAI SDK for running the LLM validation examples:

```bash
poetry install --extras llm
```

### Installing R dependencies

If you want to use the R support that is provided by the ValidMind Library, you must have R installed on your machine. You can download R from <https://cran.r-project.org/>. On a Mac, you can install R using Homebrew:

```bash
brew install r
```

Once you have R installed, install the `r-support` extra to install the necessary dependencies for R by running:

```bash
poetry install --extras r-support
```

### Versioning

Make sure you bump the package version before merging a PR with the following command:

```bash
make version tag=patch
```

The value of `tag` corresponds to one of the options provided by Poetry: <https://python-poetry.org/docs/cli/#version>

## Generating API Reference Docs

The [API reference documentation](https://docs.validmind.ai/validmind/validmind.html) you see in our docs site is generated in HTML format with `pdoc` with the following
command:

```bash
# Generate HTML
make docs
```

The resulting docs are written to `docs/pdoc/_build`.

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

## Adding a Copyright Header

When adding new files to the project, you can add the ValidMind copyright header to any files that
are missing it by running:

```bash
make copyright
```

## Known Issues

### ValidMind wheel errors

If you run into an error related to the ValidMind wheel, try:

```bash
poetry add wheel
poetry update wheel
poetry install
```
