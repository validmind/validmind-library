# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
HTML-based progress bar that preserves state in saved notebooks.
"""

import uuid

from IPython.display import HTML, display, update_display

from .html_renderer import StatefulHTMLRenderer


class HTMLProgressBar:
    """HTML-based progress bar that preserves state when notebook is saved."""

    def __init__(self, max_value: int, description: str = "Running test suite..."):
        """Initialize the progress bar.

        Args:
            max_value: Maximum value for the progress bar
            description: Initial description text
        """
        self.max_value = max_value
        self.value = 0
        self.description = description
        self.bar_id = f"progress-{uuid.uuid4().hex[:8]}"
        self._display_id = f"display-{self.bar_id}"
        self._displayed = False

    def display(self):
        """Display the progress bar."""
        if not self._displayed:
            html_content = StatefulHTMLRenderer.render_live_progress_bar(
                max_value=self.max_value,
                description=self.description,
                bar_id=self.bar_id,
            )
            display(HTML(html_content), display_id=self._display_id)
            self._displayed = True

    def update(self, value: int, description: str = None):
        """Update the progress bar value and description.

        Args:
            value: New progress value
            description: Optional new description
        """
        self.value = value
        if description:
            self.description = description

        if self._displayed:
            self._update_fallback()

    def _update_fallback(self):
        """Fallback method to update progress bar by replacing the entire HTML."""
        html_content = StatefulHTMLRenderer.render_progress_bar(
            value=self.value,
            max_value=self.max_value,
            description=self.description,
            bar_id=self.bar_id,
        )
        try:
            update_display(HTML(html_content), display_id=self._display_id)
        except Exception:
            pass

    def complete(self):
        """Mark the progress bar as complete."""
        self.update(self.max_value, "Test suite complete!")

    def close(self):
        """Close/hide the progress bar."""
        if self._displayed:
            final_html = StatefulHTMLRenderer.render_progress_bar(
                value=self.value,
                max_value=self.max_value,
                description=self.description,
                bar_id=self.bar_id,
            )
            update_display(HTML(final_html), display_id=self._display_id)


class HTMLLabel:
    """HTML-based label that preserves state when notebook is saved."""

    def __init__(self, value: str = ""):
        """Initialize the label.

        Args:
            value: Initial label text
        """
        self.value = value
        self.label_id = f"label-{uuid.uuid4().hex[:8]}"
        self._display_id = f"display-{self.label_id}"
        self._displayed = False

    def display(self):
        """Display the label."""
        if not self._displayed:
            html_content = f"""
            <div class="vm-label" id="{self.label_id}" style="font-weight: bold; margin: 5px 0;">
                {self.value}
            </div>
            """
            display(HTML(html_content), display_id=self._display_id)
            self._displayed = True

    def update(self, value: str):
        """Update the label text.

        Args:
            value: New label text
        """
        self.value = value

        if self._displayed:
            update_script = f"""
            <script>
            var labelElement = document.getElementById('{self.label_id}');
            if (labelElement) {{
                labelElement.innerHTML = {repr(value)};
            }}
            </script>
            """
            display(HTML(update_script), display_id=f"update-{self._display_id}")


class HTMLBox:
    """HTML-based container that preserves state when notebook is saved."""

    def __init__(
        self,
        children=None,
        layout_style="display: flex; align-items: center; gap: 10px;",
    ):
        """Initialize the box container.

        Args:
            children: List of child elements
            layout_style: CSS style for the container
        """
        self.children = children or []
        self.layout_style = layout_style
        self.box_id = f"box-{uuid.uuid4().hex[:8]}"
        self._display_id = f"display-{self.box_id}"
        self._displayed = False

    def display(self):
        """Display the box and its children."""
        if not self._displayed:
            child_html_parts = []
            for child in self.children:
                if hasattr(child, "display"):
                    child.display()
                if hasattr(child, "bar_id"):
                    child_html_parts.append(f'<div id="{child.bar_id}"></div>')
                elif hasattr(child, "label_id"):
                    child_html_parts.append(f'<div id="{child.label_id}"></div>')

            html_content = f"""
            <div class="vm-box" id="{self.box_id}" style="{self.layout_style}">
                {''.join(child_html_parts)}
            </div>
            """
            display(HTML(html_content), display_id=self._display_id)
            self._displayed = True
