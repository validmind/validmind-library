# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass

from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.logging import get_logger
from validmind.vm_models.model import VMModel, has_method_with_arguments

logger = get_logger(__name__)


@dataclass
class HFModel(VMModel):
    def __init__(
        self,
        input_id: str = None,
        model: object = None,
        attributes: object = None,
        name: str = None,
        **kwargs,
    ):
        super().__init__(
            input_id=input_id, model=model, attributes=attributes, name=name, **kwargs
        )

    def __post_init__(self):
        self.library = self.model.__class__.__module__.split(".")[0]
        self.class_ = self.model.__class__.__name__
        self.name = self.name or type(self.model).__name__

    def predict_proba(self, *args, **kwargs):
        """
        Invoke predict_proba from underline model
        """
        if not has_method_with_arguments(self.model, "predict_proba", 1):
            raise MissingOrInvalidModelPredictFnError(
                "Model requires a implementation of predict_proba method with 1 argument"
                + " that is tensor features matrix"
            )

        if callable(getattr(self.model, "predict_proba", None)):
            return self.model.predict_proba(*args, **kwargs)[:, 1]

    def predict(self, data):
        """
        Predict method for the model. This is a wrapper around the HF model's pipeline function
        """
        results = self.model([str(datapoint) for datapoint in data])
        tasks = self.model.__class__.__module__.split(".")

        if "text2text_generation" in tasks:
            return [result["summary_text"] for result in results]
        elif "text_classification" in tasks:
            return [result["label"] for result in results]
        elif tasks[-1] == "feature_extraction":
            # Extract [CLS] token embedding for each input and return as list of lists
            return [embedding[0][0] for embedding in results]
        else:
            return results
