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
from typing import Optional, Union

import matplotlib
import plotly.graph_objs as go

from ..client_config import client_config
from ..errors import UnsupportedFigureError
from ..logging import get_logger
from ..utils import get_full_typename
from .html_renderer import StatefulHTMLRenderer

logger = get_logger(__name__)

# Set once Plotly's static image export (kaleido + Chrome) fails, so later
# figures skip it instead of retrying (and hanging) on every figure.
_static_export_unavailable = False
_placeholder_png: Optional[bytes] = None


def _static_export_disabled() -> bool:
    """Static PNG export of Plotly figures can be turned off with
    VALIDMIND_PLOTLY_STATIC_EXPORT=false (e.g. on Databricks serverless,
    where Chrome can't run)."""
    return os.getenv("VALIDMIND_PLOTLY_STATIC_EXPORT", "true").lower() in (
        "false",
        "0",
        "no",
    )


def _get_placeholder_png() -> bytes:
    """A PNG standing in for a Plotly figure that couldn't be rendered. The
    figure's Plotly JSON is still uploaded, so the platform can render it."""
    global _placeholder_png
    if _placeholder_png is None:
        from matplotlib.figure import Figure as MplFigure

        fig = MplFigure(figsize=(6, 3))
        fig.text(
            0.5,
            0.5,
            "Static preview unavailable.\nThis is an interactive Plotly figure.",
            ha="center",
            va="center",
            color="#555555",
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        _placeholder_png = buffer.getvalue()
    return _placeholder_png


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
    title: Optional[str] = None,
) -> "Figure":
    """Create a VM Figure object from a raw figure object."""
    if is_matplotlib_figure(figure) or is_plotly_figure(figure) or is_png_image(figure):
        return Figure(key=key, figure=figure, ref_id=ref_id, title=title)

    raise ValueError(f"Unsupported figure type: {type(figure)}")


@dataclass
class Figure:
    """
    Figure objects track the schema supported by the ValidMind API.

    Attributes:
        key: Unique identifier for the figure within a test run.
        figure: The underlying figure object (matplotlib, plotly, or PNG bytes).
        ref_id: ID used to link the figure to its parent test result.
        title: Optional caption/title for the figure. When set, it is sent to
            the platform as ``metadata.caption`` and rendered by the document
            media registry as ``Figure N. <title>``.
        index: The figure's position in its test result, set when the result
            is logged and sent as ``metadata._index``. Keys change on every
            run, so the platform uses it to find the same figure in a re-run.
        variant: For a comparison test, the input and param combination that
            made the figure, sent as ``metadata._variant``. The platform
            refuses a figure at the same position from another combination.
    """

    key: str
    figure: Union[matplotlib.figure.Figure, go.Figure, go.FigureWidget, bytes]
    ref_id: str  # used to link figures to results
    title: Optional[str] = None  # caption/title used by the document media registry
    index: Optional[int] = None  # position in the test result, set when logged
    variant: Optional[str] = None  # comparison tests: the inputs/params behind it

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
            return self._get_plotly_png_bytes()
        elif is_png_image(self.figure):
            return self.figure
        else:
            raise UnsupportedFigureError(
                f"Unrecognized figure type: {get_full_typename(self.figure)}"
            )

    @property
    def has_static_image(self) -> bool:
        """False for a Plotly figure that can't be rendered to a real PNG."""
        if not is_plotly_figure(self.figure):
            return True
        return not (_static_export_unavailable or _static_export_disabled())

    def _get_plotly_png_bytes(self) -> bytes:
        """Render a Plotly figure to PNG, falling back to a placeholder when
        kaleido/Chrome isn't available so the run doesn't fail. The figure's
        JSON is uploaded either way."""
        global _static_export_unavailable

        if _static_export_unavailable or _static_export_disabled():
            return _get_placeholder_png()

        try:
            return self.figure.to_image(format="png")
        except Exception as e:
            _static_export_unavailable = True
            logger.warning(
                "Could not render Plotly figures to PNG (kaleido needs a working "
                "Chrome install): %s. Figures will be sent as Plotly JSON with a "
                "placeholder image. Set VALIDMIND_PLOTLY_STATIC_EXPORT=false to "
                "skip this attempt.",
                e,
            )
            return _get_placeholder_png()

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
        metadata = {"_ref_id": self.ref_id}
        if self.title:
            metadata["caption"] = self.title
        if self.index is not None:
            metadata["_index"] = self.index
        if self.variant is not None:
            metadata["_variant"] = self.variant

        return {
            "type": self._type,
            "key": self.key,
            "metadata": json.dumps(metadata, allow_nan=False),
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
