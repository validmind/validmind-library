# ValidMind R Package

## Installation

You can install ValidMind from CRAN:

```r
install.packages("validmind")
```

You can also install the package from GitHub using the `devtools` package:

```r
devtools::install_github("validmind/developer-framework", subdir="r/validmind")
```

Or you can install the package from source. Ensure you are in the `r/validmind` directory:

```r
devtools::install()
```

## Quick Start

You can connect to your ValidMind profile by providing the appropriate credentials:

```r
vm_r <- vm(
  api_key="<your_api_key_here>",
  api_secret="<your_api_secret_here>",
  model="<your_model_id_here>",
  python_version="<path_to_your_python_version_here>",
  api_host="https://api.prod.validmind.ai/api/v1/tracking"
)
```

## Fleshed out Example

Please see the `notebooks/code-sharing/r` folder for examples of how to use!

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
