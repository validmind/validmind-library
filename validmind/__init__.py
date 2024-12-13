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
  api_host = "https://api.dev.vm.validmind.ai/api/v1/tracking/tracking",
  api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  api_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  project = "<project-identifier>"
)
```

After you have pasted the code snippet into your development source code and executed the code, the Python Library API will register with ValidMind. You can now use the ValidMind Library to document and test your models, and to upload to the ValidMind Platform.
"""
import warnings

# Ignore Numba warnings. We are not requiring this package directly
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from .__version__ import __version__  # noqa: E402
from .api_client import init, log_metric, reload
from .client import (  # noqa: E402
    get_test_suite,
    init_dataset,
    init_model,
    init_r_model,
    preview_template,
    run_documentation_tests,
    run_test_suite,
)
from .tests.decorator import tags, tasks, test
from .vm_models.result import RawData

__all__ = [  # noqa
    "__version__",
    # Python Library API
    "datasets",
    "errors",
    "get_test_suite",
    "init",
    "init_dataset",
    "init_model",
    "init_r_model",
    "preview_template",
    "RawData",
    "reload",
    "run_documentation_tests",
    "run_test_suite",
    "tags",
    "tasks",
    "test",
    "tests",
    "test_suites",
    "vm_models",
    "unit_metrics",
    "log_metric",
]
