# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from validmind.logging import get_logger
from validmind.vm_models.model import ModelAttributes, ModelPipeline, VMModel

logger = get_logger(__name__)


class PipelineModel(VMModel):
    """
    An base class that wraps a trained model instance and its associated data.

    Attributes:
        pipeline (ModelPipeline): A pipeline of models to be executed. ModelPipeline
            is just a simple container class with a list that can be chained with the
            `|` operator.
        input_id (str, optional): The input ID for the model. Defaults to None.
        attributes (ModelAttributes, optional): The attributes of the model. Defaults to None.
        name (str, optional): The name of the model. Defaults to the class name.
    """

    predict_col: str = None

    def __init__(
        self,
        pipeline: ModelPipeline,
        attributes: ModelAttributes = None,
        input_id: str = None,
        name: str = None,
    ):
        self.pipeline = pipeline
        self.input_id = input_id

        self.language = "Python"
        self.library = self.__class__.__name__
        self.library_version = "N/A"
        self.class_ = self.__class__.__name__

        self.name = name or self.__class__.__name__

        self.attributes = attributes

    def __or__(self, other):
        if not isinstance(other, VMModel):
            raise ValueError("Can only chain VMModel objects")

        return ModelPipeline([self, other])

    def serialize(self):
        """
        Serializes the model to a dictionary so it can be sent to the API
        """
        return {
            "attributes": self.attributes.__dict__,
        }

    def predict(self, X):
        X = X.copy()

        for model in self.pipeline.models:
            predictions = model.predict(X)
            X[model.input_id] = predictions

        return predictions
