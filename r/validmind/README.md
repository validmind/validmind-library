# ValidMind R Package

## Prerequisites

Install the required R packages:

```r
install.packages(c("reticulate", "dplyr", "caTools", "knitr", "glue", "plotly", "htmltools", "rmarkdown", "DT", "base64enc"))
```

You also need a Python environment with the `validmind` Python package and `rpy2` installed:

```bash
pip install validmind rpy2
```

**Note:** On macOS, if `rpy2` fails to find R libraries, rebuild it from source against your installed R:

```bash
R_HOME=$(Rscript -e 'cat(R.home())') pip install --no-binary :all: --force-reinstall rpy2
```

Point `python_version` to your Python binary (e.g. the one in your project's `.venv`).

## Installation

You can install ValidMind from CRAN:

```r
install.packages("validmind")
```

You can also install the package from GitHub using the `devtools` package:

```r
devtools::install_github("validmind/validmind-library", subdir="r/validmind")
```

Or you can install the package from source. Ensure you are in the `r/validmind` directory:

```r
devtools::install()
```

For local development, you can skip `devtools` entirely and install directly from the repo path:

```r
install.packages("/path/to/validmind-library/r/validmind", repos = NULL, type = "source")
```

## QuickStart

You can connect to your ValidMind profile by providing the appropriate credentials:

```r
vm_r <- vm(
  api_key="<your_api_key_here>",
  api_secret="<your_api_secret_here>",
  model="<your_model_id_here>",
  python_version="<path_to_your_python_version_here>",
  api_host="https://api.prod.validmind.ai/api/v1/tracking",
  document="documentation"
)
```

The `document` parameter specifies which document type to associate with the session (e.g. `"documentation"` for model development or `"validation-report"` for model validation).

### Quickstart notebooks

See the `notebooks/code_sharing/r/` folder for full working examples:

- **`quickstart_model_documentation.Rmd`** — End-to-end model documentation workflow: load data, preprocess, train a GLM model, and run the full documentation test suite.
- **`quickstart_model_validation.Rmd`** — End-to-end model validation workflow: load data, run data quality tests, train a champion GLM model, and run model evaluation tests.

These notebooks can be run from VS Code (with the R extension), RStudio, or interactively in a terminal R session. When running interactively, launch R from the repository root so that relative dataset paths resolve correctly.

### Key APIs available via reticulate

Since the R package returns the full Python `validmind` module, you can call any Python API directly:

```r
# Preview the documentation template
vm_r$preview_template()

# Initialize datasets
vm_dataset <- vm_r$init_dataset(dataset=df, input_id="my_dataset", target_column="target")

# Initialize R models
model_path <- save_model(model)
vm_model <- vm_r$init_r_model(model_path=model_path, input_id="model")

# Assign predictions
vm_dataset$assign_predictions(model=vm_model)

# Run the full documentation test suite
vm_r$run_documentation_tests(config=test_config)

# Run individual tests
vm_r$tests$run_test("validmind.data_validation.ClassImbalance", inputs=list(dataset=vm_dataset))$log()

# List available tests
vm_r$tests$list_tests(tags=list("data_quality"), task="classification")
vm_r$tests$list_tasks_and_tags()
```

## Troubleshooting

### Initializating vm() on Mac

When calling `vm()` you might see the following error:

```
OSError: dlopen(/Users/user/validmind-sdk/.venv/lib/python3.11/site-packages/llvmlite/binding/libllvmlite.dylib, 0x0006): Library not loaded: @rpath/libc++.1.dylib
  Referenced from: <F814708F-6874-3A38-AD06-6C06514419D4> /Users/user/validmind-sdk/.venv/lib/python3.11/site-packages/llvmlite/binding/libllvmlite.dylib
  Reason: tried: '/Library/Frameworks/R.framework/Resources/lib/libc++.1.dylib' (no such file), '/Library/Java/JavaVirtualMachines/jdk-11.0.18+10/Contents/Home/lib/server/libc++.1.dylib' (no such file), '/var/folders/c4/typylth55knbkn7qjm8zd0jr0000gn/T/rstudio-fallback-library-path-492576811/libc++.1.dylib' (no such file)
```

This is typically due to the `libc++` library not being found but it's possible that is already installed and R cannot find it. You can solve this by finding the path to the library and creating a symlink to it.

```
# Find the path to libc++.1.dylib. This can return multiple results.
sudo find / -name "libc++.1.dylib" 2>/dev/null
```

If you are using Homebrew, the command above will return a path like `/opt/homebrew/Cellar/llvm/...`. You can create a symlink to the library by running the following command:

```
sudo ln -s <path_to_libc++.1.dylib> /Library/Frameworks/R.framework/Resources/lib/libc++.1.dylib
```

Note that the target argument in the path of `libc++` that R was trying to find.

After creating the symlink, you can try calling `vm()` again.

### Issues with Numba when initializing vm() on Mac

You might also see the following error when initializing vm():

```
Error in py_module_import(module, convert = convert) :
  ImportError: cannot import name 'NumbaTypeError' from partially initialized module 'numba.core.errors' (most likely due to a circular import) (/Users/user/validmind-sdk/.venv/lib/python3.11/site-packages/numba/core/errors.py)
```

To fix this, you can reinstall Numba:

```
pip install -U numba
```

And restart the R session.
