# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
The ValidMind Library is a suite of developer tools and methods designed to automate the documentation and validation of your models.

Designed to be model agnostic, the ValidMind Library provides all the standard functionality without requiring you to rewrite any functions as long as your model is built in Python.

With a rich array of documentation tools and test suites, from documenting descriptions of your datasets to testing your models for weak spots and overfit areas, the ValidMind Library helps you automate model documentation by feeding the ValidMind Platform with documentation artifacts and test results.

To install the ValidMind Library:

```bash
pip install validmind
```

To initialize the ValidMind Library, paste the code snippet with the model identifier credentials directly into your development source code, replacing this example with your own:

```python
import validmind as vm

vm.init(
  api_host = "https://app.prod.validmind.ai/api/v1/tracking/tracking",
  api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  api_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  project = "<project-identifier>"
)
```

After you have pasted the code snippet into your development source code and executed the code, the Python Library API will register with ValidMind. You can now use the ValidMind Library to document and test your models, and to upload to the ValidMind Platform.
"""
import threading
import warnings
from importlib import metadata

from IPython.display import HTML, display

# Ignore Numba warnings. We are not requiring this package directly
try:
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPendingDeprecationWarning,
    )

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
except ImportError:
    ...

from . import scorer
from .__version__ import __version__  # noqa: E402
from .api_client import init, log_metric, log_text, reload
from .client import (  # noqa: E402
    get_test_suite,
    init_dataset,
    init_model,
    init_r_model,
    preview_template,
    run_documentation_tests,
    run_test_suite,
)
from .experimental import agents as experimental_agent
from .tests.decorator import scorer as scorer_decorator
from .tests.decorator import tags, tasks, test
from .tests.run import print_env
from .utils import is_notebook, parse_version
from .vm_models.result import RawData

__shown = False


def show_warning(installed, running):
    global __shown

    if __shown:
        return
    __shown = True

    message = (
        f"⚠️ This kernel is running an older version of validmind ({running}) "
        f"than the latest version installed on your system ({installed}).\n\n"
        "You may need to restart the kernel if you are experiencing issues."
    )
    display(HTML(f"<div style='color: red;'>{message}</div>"))


def check_version():
    # get the installed vs running version of validmind
    # to make sure we are using the latest installed version
    # in case user has updated the package but forgot to restart the kernel
    try:
        installed = metadata.version("validmind")
    except metadata.PackageNotFoundError:
        # Package metadata not found, skip version check
        return

    running = __version__

    if parse_version(installed) > parse_version(running):
        show_warning(installed, running)

    # Schedule the next check for 5 minutes from now
    timer = threading.Timer(300, check_version)
    timer.daemon = True
    timer.start()


if is_notebook():
    check_version()

__all__ = [  # noqa
    "__version__",
    # main library API
    "init",
    "init_dataset",
    "init_model",
    "init_r_model",
    "get_test_suite",
    "log_metric",
    "preview_template",
    "print_env",
    "reload",
    "run_documentation_tests",
    # log metric function (for direct/bulk/retroactive logging of metrics)
    # test suite functions (less common)
    "run_test_suite",
    # helper functions (for troubleshooting)
    # decorators (for building tests
    "tags",
    "tasks",
    "test",
    "scorer_decorator",
    # scorer module
    "scorer",
    # raw data (for post-processing test results and building tests)
    "RawData",
    # submodules
    "datasets",
    "errors",
    "vm_models",
    "tests",
    "unit_metrics",
    "test_suites",
    "log_text",
    # experimental features
    "experimental_agent",
]
