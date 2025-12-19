# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Figure objects track the figure schema supported by the ValidMind API.
"""

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Union

import matplotlib
import plotly.graph_objs as go

from ..client_config import client_config
from ..errors import UnsupportedFigureError
from ..utils import get_full_typename
from .html_renderer import StatefulHTMLRenderer


def is_matplotlib_figure(figure) -> bool:
    return isinstance(figure, matplotlib.figure.Figure)


def is_plotly_figure(figure) -> bool:
    return isinstance(figure, (go.Figure, go.FigureWidget))


def is_png_image(figure) -> bool:
    return isinstance(figure, bytes)


def create_figure(
    figure: Union[matplotlib.figure.Figure, go.Figure, go.FigureWidget, bytes],
    key: str,
    ref_id: str,
) -> "Figure":
    """Create a VM Figure object from a raw figure object."""
    if is_matplotlib_figure(figure) or is_plotly_figure(figure) or is_png_image(figure):
        return Figure(key=key, figure=figure, ref_id=ref_id)

    raise ValueError(f"Unsupported figure type: {type(figure)}")


@dataclass
class Figure:
    """
    Figure objects track the schema supported by the ValidMind API.
    """

    key: str
    figure: Union[matplotlib.figure.Figure, go.Figure, go.FigureWidget, bytes]
    ref_id: str  # used to link figures to results

    _type: str = "plot"  # for now this is the only figure type

    def __post_init__(self):
        # Wrap around with FigureWidget so that we can display interactive Plotly
        # plots in regular Jupyter notebooks. This is not supported on Google Colab.
        if (
            not client_config.running_on_colab
            and self.figure
            and is_plotly_figure(self.figure)
        ):
            self.figure = go.FigureWidget(self.figure)

    def __repr__(self):
        return f"Figure(key={self.key}, ref_id={self.ref_id})"

    def to_html(self):
        """
        Returns HTML representation that preserves state when notebook is saved.
        This is the preferred method for displaying figures in notebooks.
        """
        metadata = {"key": self.key, "ref_id": self.ref_id, "type": self._type}

        if is_matplotlib_figure(self.figure):
            tmpfile = BytesIO()
            self.figure.savefig(tmpfile, format="png")
            encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
            return StatefulHTMLRenderer.render_figure(encoded, self.key, metadata)

        elif is_plotly_figure(self.figure):
            png_file = self.figure.to_image(format="png")
            encoded = base64.b64encode(png_file).decode("utf-8")
            # Add plotly-specific metadata only if interactive figures are enabled
            if os.getenv("VALIDMIND_INTERACTIVE_FIGURES", "true").lower() in (
                "true",
                "1",
                "yes",
            ):
                metadata["plotly_json"] = self.figure.to_json()
            return StatefulHTMLRenderer.render_figure(encoded, self.key, metadata)

        elif is_png_image(self.figure):
            encoded = base64.b64encode(self.figure).decode("utf-8")
            return StatefulHTMLRenderer.render_figure(encoded, self.key, metadata)

        else:
            raise UnsupportedFigureError(
                f"Figure type {type(self.figure)} not supported for plotting"
            )

    def serialize(self):
        """
        Serializes the Figure to a dictionary so it can be sent to the API.
        """
        return {
            "type": self._type,
            "key": self.key,
            "metadata": json.dumps({"_ref_id": self.ref_id}, allow_nan=False),
        }

    def _get_b64_url(self):
        """
        Returns a base64 encoded URL for the figure.
        """
        if is_matplotlib_figure(self.figure):
            buffer = BytesIO()
            self.figure.savefig(buffer, format="png")
            buffer.seek(0)

            b64_data = base64.b64encode(buffer.read()).decode("utf-8")

            return f"data:image/png;base64,{b64_data}"

        elif is_plotly_figure(self.figure):
            bytes = self.figure.to_image(format="png")
            b64_data = base64.b64encode(bytes).decode("utf-8")

            return f"data:image/png;base64,{b64_data}"

        elif is_png_image(self.figure):
            b64_data = base64.b64encode(self.figure).decode("utf-8")

            return f"data:image/png;base64,{b64_data}"

        raise UnsupportedFigureError(
            f"Unrecognized figure type: {get_full_typename(self.figure)}"
        )

    def serialize_files(self):
        """Creates a `requests`-compatible files object to be sent to the API."""
        if is_matplotlib_figure(self.figure):
            buffer = BytesIO()
            self.figure.savefig(buffer, bbox_inches="tight")
            buffer.seek(0)
            return {"image": (f"{self.key}.png", buffer, "image/png")}

        elif is_plotly_figure(self.figure):
            # When using plotly, we need to use we will produce two files:
            # - a JSON file that will be used to display the figure in the ValidMind Platform
            # - a PNG file that will be used to display the figure in documents
            return {
                "image": (
                    f"{self.key}.png",
                    self.figure.to_image(format="png"),
                    "image/png",
                ),
                "json_image": (
                    f"{self.key}.json",
                    self.figure.to_json(),
                    "application/json",
                ),
            }

        elif is_png_image(self.figure):
            return {"image": (f"{self.key}.png", self.figure, "image/png")}

        raise UnsupportedFigureError(
            f"Unrecognized figure type: {get_full_typename(self.figure)}"
        )
