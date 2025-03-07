# ValidMind Library

The ValidMind Library is a suite of developer tools and methods designed to automate the documentation and validation of your models.

Designed to be model agnostic, the ValidMind Library provides all the standard functionality without requiring you to rewrite any functions as long as your model is built in Python.

With a rich array of documentation tools and test suites, from documenting descriptions of your datasets to testing your models for weak spots and overfit areas, the ValidMind Library helps you automate model documentation by feeding the ValidMind Platform with documentation artifacts and test results.

## What is ValidMind?

ValidMind helps developers, data scientists and risk and compliance stakeholders identify potential risks in their AI and large language models, and generate robust, high-quality model documentation that meets regulatory requirements.

[The ValidMind AI risk platform](https://docs.validmind.ai/about/overview.html) consists of two intertwined product offerings:

- **The ValidMind Library** — Designed to be incorporated into your existing model development environment, you use the ValidMind Library to run tests and log documentation to the ValidMind Platform. Driven by the power of open-source, the ValidMind Library welcomes contributions to our code and developer samples: [`validmind-library` @ GitHub](https://github.com/validmind/validmind-library)
- **The ValidMind Platform** — A cloud-hosted user interface allowing you to comprehensively track your model inventory throughout the entire model lifecycle according to the unique requirements of your organization. You use the ValidMind Platform to oversee your model risk management process via the customizable model inventory.

### What do I need to get started with ValidMind?

> **All you need to get started with ValidMind is an account with us.**
>
> Signing up is FREE — **[Register with ValidMind](https://docs.validmind.ai/guide/configuration/register-with-validmind.html)**

That's right — you can run tests and log documentation even if you don't have a model available, so go ahead and get started with the [**ValidMind Library**](https://docs.validmind.ai/developer/validmind-library.html)!

### How do I do more with the ValidMind Library?

**[Explore our code samples!](https://docs.validmind.ai/developer/samples-jupyter-notebooks.html)**

Our selection of Jupyter Notebooks showcase the capabilities and features of the ValidMind Library, while also providing you with useful examples that you can build on and adapt for your own use cases.

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
    pip install validmind
    pip install rpy2
    ```
