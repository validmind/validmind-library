# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Base class for ValidMind Input types"""

from abc import ABC
from typing import Any, Dict


class VMInput(ABC):
    """
    Base class for ValidMind Input types.
    """

    def with_options(self, **kwargs: Dict[str, Any]) -> "VMInput":
        """
        Allows for setting options on the input object that are passed by the user
        when using the input to run a test or set of tests.

        To allow options, just override this method in the subclass (see VMDataset)
        and ensure that it returns a new instance of the input with the specified options
        set.

        Args:
            **kwargs: Arbitrary keyword arguments that will be passed to the input object.

        Returns:
            VMInput: A new instance of the input with the specified options set.
        """
        if kwargs:
            raise NotImplementedError("This type of input does not support options")
