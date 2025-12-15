# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
HTML renderer for ValidMind components that preserves state in saved notebooks.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class StatefulHTMLRenderer:
    """Renders ValidMind components as self-contained HTML with embedded state."""

    # Plotly.js CDN URL - using a stable version
    PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.27.0.min.js"

    @staticmethod
    def render_figure(
        figure_data: str, key: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render a figure as HTML with embedded data.

        For Plotly figures, renders an interactive chart with the static image
        as a fallback for environments without JavaScript support.

        Args:
            figure_data: Base64-encoded image data
            key: Unique key for the figure
            metadata: Optional metadata to embed (may contain plotly_json for
                      interactive Plotly rendering)

        Returns:
            HTML string with embedded figure and metadata
        """
        metadata = metadata or {}
        plotly_json = metadata.get("plotly_json")

        # Create a copy of metadata without plotly_json for the embedded metadata
        # (to avoid duplicating the large JSON in the HTML)
        metadata_for_embed = {k: v for k, v in metadata.items() if k != "plotly_json"}
        metadata_for_embed["has_plotly_json"] = plotly_json is not None
        metadata_json = json.dumps(metadata_for_embed, default=str)

        # Static image HTML (used as fallback or primary display)
        img_html = f"""<img id="vm-img-{key}" src="data:image/png;base64,{figure_data}"
                 style="width:100%; height: auto; max-width: 800px;"
                 alt="ValidMind Figure {key}"/>"""

        if plotly_json:
            plotly_cdn_url = StatefulHTMLRenderer.PLOTLY_CDN_URL

            # Render with static image visible by default, JavaScript upgrades to interactive
            # This ensures the image shows even when scripts are blocked (e.g., Google Colab)
            return f"""
        <div class="vm-figure" data-key="{key}" id="figure-{key}">
            <div id="vm-plotly-{key}" style="width:100%; max-width: 800px; display: none;"></div>
            <div id="vm-img-container-{key}">{img_html}</div>
            <script type="application/json" class="vm-metadata">{metadata_json}</script>
            <script type="application/json" class="vm-plotly-data">{plotly_json}</script>
            <script>
            (function() {{
                var plotlyContainer = document.getElementById('vm-plotly-{key}');
                var imgContainer = document.getElementById('vm-img-container-{key}');
                var dataScript = plotlyContainer.parentElement.querySelector('.vm-plotly-data');
                if (!dataScript || !plotlyContainer || !imgContainer) return;

                function renderPlotly() {{
                    try {{
                        var plotData = JSON.parse(dataScript.textContent);
                        var layout = plotData.layout || {{}};
                        layout.autosize = true;
                        Plotly.newPlot('vm-plotly-{key}', plotData.data, layout, {{responsive: true}});
                        // Only hide image and show plotly after successful render
                        imgContainer.style.display = 'none';
                        plotlyContainer.style.display = 'block';
                    }} catch (e) {{
                        console.error('Failed to render Plotly chart:', e);
                        // Keep showing the static image (already visible)
                    }}
                }}

                if (typeof Plotly !== 'undefined') {{
                    renderPlotly();
                }} else {{
                    var script = document.createElement('script');
                    script.src = '{plotly_cdn_url}';
                    script.onload = renderPlotly;
                    // On error, static image remains visible (no action needed)
                    document.head.appendChild(script);
                }}
            }})();
            </script>
        </div>
        """
        else:
            # Non-Plotly figures (matplotlib, PNG) - render static image only
            return f"""
        <div class="vm-figure" data-key="{key}" id="figure-{key}">
            {img_html}
            <script type="application/json" class="vm-metadata">{metadata_json}</script>
        </div>
        """

    @staticmethod
    def render_table(
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        title: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> str:
        """Render a table as HTML.

        Args:
            data: DataFrame or list of dictionaries
            title: Optional table title
            table_id: Optional unique ID for the table

        Returns:
            HTML string with table
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)

        if table_id is None:
            table_id = f"table-{uuid.uuid4().hex[:8]}"

        title_html = f"<h4>{title}</h4>" if title else ""

        # Convert DataFrame to HTML with styling
        table_html = data.to_html(
            classes="vm-table table table-striped table-hover",
            table_id=table_id,
            escape=False,
            index=False,
        )

        return f"""
        <div class="vm-table-container" id="{table_id}-container">
            {title_html}
            {table_html}
        </div>
        """

    @staticmethod
    def render_accordion(
        items: List[str], titles: List[str], accordion_id: Optional[str] = None
    ) -> str:
        """Render an accordion component as HTML with JavaScript.

        Args:
            items: List of HTML content for each accordion item
            titles: List of titles for each accordion item
            accordion_id: Optional unique ID for the accordion

        Returns:
            HTML string with accordion and embedded JavaScript
        """
        if accordion_id is None:
            accordion_id = f"accordion-{uuid.uuid4().hex[:8]}"

        accordion_items = []

        for i, (title, content) in enumerate(zip(titles, items)):
            item_id = f"{accordion_id}-item-{i}"
            accordion_items.append(
                f"""
            <div class="vm-accordion-item">
                <div class="vm-accordion-header"
                     onclick="toggleAccordionItem('{item_id}')"
                     style="cursor: pointer; padding: 10px; background-color: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold;">
                    <span class="vm-accordion-toggle" id="{item_id}-toggle">▶</span>
                    {title}
                </div>
                <div class="vm-accordion-content"
                     id="{item_id}"
                     style="display: none; padding: 15px; border: 1px solid #dee2e6; border-top: none;">
                    {content}
                </div>
            </div>
            """
            )

        return f"""
        <div class="vm-accordion" id="{accordion_id}">
            {''.join(accordion_items)}
        </div>

        <script>
        function toggleAccordionItem(itemId) {{
            const content = document.getElementById(itemId);
            const toggle = document.getElementById(itemId + '-toggle');

            if (content.style.display === 'none' || content.style.display === '') {{
                content.style.display = 'block';
                toggle.innerHTML = '▼';
            }} else {{
                content.style.display = 'none';
                toggle.innerHTML = '▶';
            }}
        }}
        </script>
        """

    @staticmethod
    def render_progress_bar(
        value: int, max_value: int, description: str = "", bar_id: Optional[str] = None
    ) -> str:
        """Render a progress bar as HTML.

        Args:
            value: Current progress value
            max_value: Maximum value
            description: Progress description
            bar_id: Optional unique ID for the progress bar

        Returns:
            HTML string with progress bar
        """
        if bar_id is None:
            bar_id = f"progress-{uuid.uuid4().hex[:8]}"

        percentage = (value / max_value * 100) if max_value > 0 else 0

        return f"""
        <div class="vm-progress-container" id="{bar_id}">
            <div class="vm-progress-description">{description}</div>
            <div class="vm-progress-bar"
                 style="width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
                <div class="vm-progress-fill"
                     style="width: {percentage}%; height: 20px; background-color: #007bff; transition: width 0.3s ease;">
                </div>
            </div>
            <div class="vm-progress-text">{value}/{max_value} ({percentage:.1f}%)</div>
        </div>
        """

    @staticmethod
    def render_live_progress_bar(
        max_value: int,
        description: str = "Running test suite...",
        bar_id: Optional[str] = None,
    ) -> str:
        """Render a live-updating progress bar as HTML with JavaScript.

        Args:
            max_value: Maximum value for the progress bar
            description: Initial description text
            bar_id: Optional unique ID for the progress bar

        Returns:
            HTML string with live progress bar and update functions
        """
        if bar_id is None:
            bar_id = f"progress-{uuid.uuid4().hex[:8]}"

        return f"""
        <div class="vm-progress-container" id="{bar_id}">
            <div class="vm-progress-description" id="{bar_id}-description">{description}</div>
            <div class="vm-progress-bar"
                 style="width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden; margin: 10px 0;">
                <div class="vm-progress-fill" id="{bar_id}-fill"
                     style="width: 0%; height: 20px; background-color: #007bff; transition: width 0.3s ease;">
                </div>
            </div>
            <div class="vm-progress-text" id="{bar_id}-text">0/{max_value} (0.0%)</div>
        </div>

        <script>
        // Create global update functions for this progress bar
        window.updateProgress_{bar_id.replace('-', '_')} = function(value, description) {{
            var maxValue = {max_value};
            var percentage = (value / maxValue * 100);

            var fillElement = document.getElementById('{bar_id}-fill');
            var textElement = document.getElementById('{bar_id}-text');
            var descElement = document.getElementById('{bar_id}-description');

            if (fillElement) fillElement.style.width = percentage + '%';
            if (textElement) textElement.innerHTML = value + '/' + maxValue + ' (' + percentage.toFixed(1) + '%)';
            if (descElement && description) descElement.innerHTML = description;
        }};

        window.completeProgress_{bar_id.replace('-', '_')} = function() {{
            var descElement = document.getElementById('{bar_id}-description');
            if (descElement) descElement.innerHTML = 'Test suite complete!';
        }};
        </script>
        """

    @staticmethod
    def render_result_header(
        test_name: str,
        passed: Optional[bool] = None,
        metric: Optional[Union[int, float]] = None,
    ) -> str:
        """Render a test result header.

        Args:
            test_name: Name of the test
            passed: Whether the test passed (None for no status)
            metric: Optional metric value

        Returns:
            HTML string with result header
        """
        if passed is None:
            status_icon = ""
        else:
            status_icon = "✅" if passed else "❌"

        metric_html = f": <code>{metric}</code>" if metric is not None else ""

        return f"""
        <div class="vm-result-header">
            <h3>{status_icon} {test_name}{metric_html}</h3>
        </div>
        """

    @staticmethod
    def render_description(description: str) -> str:
        """Render a description with proper formatting.

        Args:
            description: Description text (may contain HTML)

        Returns:
            HTML string with formatted description
        """
        formatted_description = description.replace("<h3>", "<strong>").replace(
            "</h3>", "</strong>"
        )

        return f"""
        <div class="vm-description result">
            {formatted_description}
        </div>
        """

    @staticmethod
    def render_parameters(params: Dict[str, Any]) -> str:
        """Render parameters as formatted JSON.

        Args:
            params: Parameters dictionary

        Returns:
            HTML string with formatted parameters
        """
        params_json = json.dumps(params, indent=2, default=str)

        return f"""
        <div class="vm-parameters">
            <h4>Parameters:</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;">
<code>{params_json}</code>
            </pre>
        </div>
        """

    @staticmethod
    def get_base_css() -> str:
        """Get base CSS styles for ValidMind HTML components.

        Returns:
            CSS string with base styles
        """
        return """
        <style>
        .vm-result {
            margin: 10px 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .vm-description.result {
            margin: 10px 0;
            padding: 10px;
            background-color: #f1f1f1;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        .vm-table {
            width: 100%;
            margin: 10px 0;
            border-collapse: collapse;
        }

        .vm-table th,
        .vm-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .vm-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }

        .vm-table tr:hover {
            background-color: #f5f5f5;
        }

        .vm-figure {
            margin: 15px 0;
            text-align: center;
        }

        .vm-accordion-item {
            margin: 5px 0;
        }

        .vm-accordion-header:hover {
            background-color: #e9ecef !important;
        }

        .vm-progress-container {
            margin: 10px 0;
        }

        .vm-progress-description {
            margin-bottom: 5px;
            font-weight: bold;
        }

        .vm-progress-text {
            margin-top: 5px;
            font-size: 0.9em;
            color: #666;
        }

        .vm-result-header h3 {
            margin: 10px 0;
            color: #333;
        }

        .vm-parameters {
            margin: 15px 0;
        }

        .vm-parameters h4 {
            margin-bottom: 10px;
            color: #333;
        }
        </style>
        """
