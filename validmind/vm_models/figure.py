# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
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
    _cached_png_bytes: bytes = None  # cached PNG bytes for async-safe serialization

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

    def _get_png_bytes(self) -> bytes:
        """Get PNG bytes for the figure, using cache if available.

        This method returns cached PNG bytes if pre_serialize() was called,
        otherwise it generates them on demand. Caching is important for Plotly
        figures because to_image() uses kaleido which conflicts with asyncio
        event loops when called from within an async context.
        """
        if self._cached_png_bytes is not None:
            return self._cached_png_bytes

        if is_matplotlib_figure(self.figure):
            buffer = BytesIO()
            self.figure.savefig(buffer, format="png")
            buffer.seek(0)
            return buffer.read()
        elif is_plotly_figure(self.figure):
            return self.figure.to_image(format="png")
        elif is_png_image(self.figure):
            return self.figure
        else:
            raise UnsupportedFigureError(
                f"Unrecognized figure type: {get_full_typename(self.figure)}"
            )

    def pre_serialize(self):
        """Pre-serialize the figure to PNG bytes for async-safe upload.

        Call this method before entering an async context (e.g., before run_async)
        to avoid conflicts between Plotly's kaleido library and asyncio event loops.
        The cached bytes will be used by to_html(), _get_b64_url(), and serialize_files().
        """
        if self._cached_png_bytes is None:
            self._cached_png_bytes = self._get_png_bytes()

    def to_html(self):
        """
        Returns HTML representation that preserves state when notebook is saved.
        This is the preferred method for displaying figures in notebooks.
        """
        metadata = {"key": self.key, "ref_id": self.ref_id, "type": self._type}

        png_bytes = self._get_png_bytes()
        encoded = base64.b64encode(png_bytes).decode("utf-8")

        # Add plotly-specific metadata only if interactive figures are enabled
        if is_plotly_figure(self.figure) and os.getenv(
            "VALIDMIND_INTERACTIVE_FIGURES", "true"
        ).lower() in ("true", "1", "yes"):
            metadata["plotly_json"] = self.figure.to_json()

        return StatefulHTMLRenderer.render_figure(encoded, self.key, metadata)

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
        png_bytes = self._get_png_bytes()
        b64_data = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def serialize_files(self):
        """Creates a `requests`-compatible files object to be sent to the API."""
        png_bytes = self._get_png_bytes()
        files = {"image": (f"{self.key}.png", png_bytes, "image/png")}

        # For Plotly figures, also include the JSON for interactive display
        if is_plotly_figure(self.figure):
            files["json_image"] = (
                f"{self.key}.json",
                self.figure.to_json(),
                "application/json",
            )

        return files
