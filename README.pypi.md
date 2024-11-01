# ValidMind Library

ValidMind's Python library automates the documentation and validation of your models through a comprehensive suite of developer tools and methods.

Built to be model agnostic, it works seamlessly with any Python model without requiring developers to rewrite existing code.

It includes a suite of rich documentation and model testing capabilities - from dataset descriptions to identifying model weak spots and overfit areas. Through this library, you can automate documentation generation by feeding artifacts and test results to the ValidMind platform.

## Installation

To install the library and all optional dependencies, run:

```bash
pip install validmind[all]
```

To install the library without optional dependencies (core functionality only), run:

```bash
pip install validmind
```

### Extra dependencies

The library has optional dependencies that can be installed separately to support additional model types and tests.

- **LLM Support**: To be able to run tests for Large Language Models (LLMs), install the `llm` extra:

    ```bash
    pip install validmind[llm]
    ```

- **PyTorch Models**: To use pytorch models with the library, install the `torch` extra:

    ```bash
    pip install validmind[torch]
    ```

- **Hugging Face Transformers**: To use Hugging Face Transformers models with the library, install the `transformers` extra:

    ```bash
    pip install validmind[transformers]
    ```

- **R Models**: To use R models with the library, install the `r` extra:

    ```bash
    pip install validmind[r-support]
    ```
