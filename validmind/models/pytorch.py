# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.logging import get_logger
from validmind.vm_models.model import VMModel, has_method_with_arguments

logger = get_logger(__name__)


class PyTorchModel(VMModel):
    """PyTorchModel class wraps a PyTorch model"""

    def __post_init__(self):
        if not self.model:
            raise ValueError("Model object is a required argument for PyTorchModel")

        self.library = "torch"
        self.name = self.name or "PyTorch Neural Network"
        self.device_type = next(self.model.parameters()).device

    def predict_proba(self, *args, **kwargs):
        """
        Invoke predict_proba from underline model
        """
        if not has_method_with_arguments(self.model, "predict_proba", 1):
            raise MissingOrInvalidModelPredictFnError(
                "Model requires a implemention of predict_proba method with 1 argument"
                + " that is tensor features matrix"
            )

        if callable(getattr(self.model, "predict_proba", None)):
            return self.model.predict_proba(*args, **kwargs)[:, 1]

    def predict(self, *args, **kwargs):
        """
        Predict method for the model. This is a wrapper around the model's
        """
        if not has_method_with_arguments(self.model, "predict", 1):
            raise MissingOrInvalidModelPredictFnError(
                "Model requires a implemention of predict method with 1 argument"
                + " that is tensor features matrix"
            )
        import torch

        return self.model.predict(torch.tensor(args[0]).to(self.device_type))
