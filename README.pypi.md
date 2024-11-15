# ValidMind Library

The ValidMind Library is a suite of developer tools and methods designed to automate the documentation and validation of your models.

Designed to be model agnostic, the ValidMind Library provides all the standard functionality without requiring you to rewrite any functions as long as your model is built in Python.

With a rich array of documentation tools and test suites, from documenting descriptions of your datasets to testing your models for weak spots and overfit areas, the ValidMind Library helps you automate model documentation by feeding the ValidMind Platform with documentation artifacts and test results. 

## Installation

To install the ValidMind Library and all optional dependencies, run:

```bash
pip install validmind[all]
```

To install the ValidMind Library without optional dependencies (core functionality only), run:

```bash
pip install validmind
```

### Extra dependencies

The ValidMind Library has optional dependencies that can be installed separately to support additional model types and tests.

- **LLM Support**: To be able to run tests for Large Language Models (LLMs), install the `llm` extra:

    ```bash
    pip install validmind[llm]
    ```

- **PyTorch Models**: To use pytorch models with the ValidMind Library, install the `torch` extra:

    ```bash
    pip install validmind[torch]
    ```

- **Hugging Face Transformers**: To use Hugging Face Transformers models with the ValidMind Library, install the `transformers` extra:

    ```bash
    pip install validmind[transformers]
    ```

- **R Models**: To use R models with the ValidMind Library, install the `r` extra:

    ```bash
    pip install validmind[r-support]
    ```
