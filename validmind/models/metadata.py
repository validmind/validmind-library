# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.vm_models.model import VMModel


class MetadataModel(VMModel):
    """
    MetadataModel is designed to represent a model that is not available for inference
    for various reasons but for which metadata and pre-computed predictions are available.

    Model attributes are required since this will be the only information we can
    collect and log about the model.

    This class should not be instantiated directly. Instead call `vm.init_model()` and
    pass in a dictionary with the required metadata as `attributes`.

    Attributes:
        attributes (ModelAttributes): The attributes of the model. Required.
        input_id (str, optional): The input ID for the model. Defaults to None.
        name (str, optional): The name of the model. Defaults to the class name.
    """

    def __post_init__(self):
        if not hasattr(self, "attributes"):
            raise ValueError("MetadataModel requires attributes")

        self.name = self.name or "Metadata Model"

    def predict(self, *args, **kwargs):
        """Not implemented for MetadataModel"""
        raise MissingOrInvalidModelPredictFnError(
            "MetadataModel does not support inference"
        )

    def predict_proba(self, *args, **kwargs):
        """Not implemented for MetadataModel"""
        raise MissingOrInvalidModelPredictFnError(
            "MetadataModel does not support inference"
        )
