# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass

import pandas as pd

from validmind.logging import get_logger
from validmind.models.function import FunctionModel

logger = get_logger(__name__)


@dataclass
class Prompt:
    template: str
    variables: list = None


class FoundationModel(FunctionModel):
    """FoundationModel class wraps a Foundation LLM endpoint

    This class wraps a predict function that is user-defined and adapts it to works
    with ValidMind's model interface for the purpose of model eval and documentation

    Attributes:
        predict_fn (callable): The predict function that should take a prompt as input
                and return the result from the model
        prompt (Prompt): The prompt object that defines the prompt template and the
                variables (if any)
        name (str, optional): The name of the model. Defaults to name of the predict_fn
    """

    def __post_init__(self):
        super().__post_init__()

        if not hasattr(self, "prompt") or not isinstance(self.prompt, Prompt):
            raise ValueError("FoundationModel requires a Prompt object")

    def _build_prompt(self, x: pd.DataFrame):
        """
        Builds the prompt for the model
        """
        return (
            self.prompt.template.format(
                **{key: x[key] for key in self.prompt.variables}
            )
            if self.prompt.variables
            else self.prompt.template
        )

    def predict(self, X: pd.DataFrame):
        """
        Predict method for the model. This is a wrapper around the model's
        """
        return [self.predict_fn(self._build_prompt(x[1])) for x in X.iterrows()]
