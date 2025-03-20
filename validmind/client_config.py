# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Central class to track configuration of the ValidMind Library
client against the ValidMind API
"""

from dataclasses import dataclass


@dataclass
class ClientConfig:
    """
    Configuration class for the ValidMind API client. This class is instantiated
    when initializing the API client.
    """

    model: object
    feature_flags: dict
    document_type: str
    documentation_template: object
    running_on_colab: bool = False

    def __post_init__(self):
        """
        Set additional attributes when initializing the class.
        """
        # check if running on notebook and set running_on_colab
        try:
            from google.colab import drive  # noqa

            self.running_on_colab = True
        except ImportError:
            self.running_on_colab = False

    def can_generate_llm_test_descriptions(self):
        """Returns True if the client can generate LLM-based test descriptions."""
        return self.feature_flags.get("llm_test_descriptions", True)


client_config = ClientConfig(
    model=None,
    feature_flags={},
    document_type="model_documentation",
    documentation_template=None,
    running_on_colab=False,
)
