# ValidMind Library

ValidMind's Python library automates the documentation and validation of your models through a comprehensive suite of developer tools and methods.

Built to be model agnostic, it works seamlessly with any Python model without requiring developers to rewrite existing code.

It includes a suite of rich documentation and model testing capabilities - from dataset descriptions to identifying model weak spots and overfit areas. Through this library, you can automate documentation generation by feeding artifacts and test results to the ValidMind platform.

## Getting started

### Install from PyPI

To install the library and all optional dependencies, run:

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
