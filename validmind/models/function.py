# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

from validmind.vm_models.model import VMModel


# semi-immutable dict
class Input(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._new = set()

    def __setitem__(self, key, value):
        self._new.add(key)
        super().__setitem__(key, value)

    def __delitem__(self, _):
        raise TypeError("Cannot delete keys from Input")

    def get_new(self) -> Dict[str, Any]:
        """Get the newly added key-value pairs.

        Returns:
            Dict[str, Any]: Dictionary containing only the newly added key-value pairs.
        """
        return {k: self[k] for k in self._new}


class FunctionModel(VMModel):
    """
    FunctionModel class wraps a user-defined predict function

    Attributes:
        predict_fn (callable): The predict function that should take a dictionary of
            input features and return a prediction. Can return simple values or
            dictionary objects.
        input_id (str, optional): The input ID for the model. Defaults to None.
        name (str, optional): The name of the model. Defaults to the name of the predict_fn.
        prompt (Prompt, optional): If using a prompt, the prompt object that defines the template
            and the variables (if any). Defaults to None.
    """

    def __post_init__(self):
        if not hasattr(self, "predict_fn") or not callable(self.predict_fn):
            raise ValueError("FunctionModel requires a callable predict_fn")

        self.name = self.name or self.predict_fn.__name__

    def predict(self, X) -> List[Any]:
        """Compute predictions for the input (X)

        Args:
            X (pandas.DataFrame): The input features to predict on

        Returns:
            List[Any]: The predictions. Can contain simple values or dictionary objects
                       depending on what the predict_fn returns.
        """
        predictions = []
        for x in X.to_dict(orient="records"):
            result = self.predict_fn(x)
            # Handle both simple values and complex dictionary returns
            predictions.append(result)

        return predictions
