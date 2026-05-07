# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils import display, md_to_html
from .html_renderer import StatefulHTMLRenderer
from .result import TextGenerationResult


@dataclass
class DocumentationTextSummary:
    """Notebook-friendly summary for generated documentation text."""

    title: str
    description: str
    results: Dict[str, TextGenerationResult]
    template_sections: Optional[List[Dict[str, Any]]] = None
    show_link: bool = True

    def display(self):
        """Display the summary."""
        display(self.to_html())

    def _get_section_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            section["id"]: section
            for section in self.template_sections or []
            if section.get("id")
        }

    def _get_top_level_section_id(self, section_id: str) -> str:
        section_map = self._get_section_map()
        current_section_id = section_id

        while current_section_id in section_map:
            parent_section_id = section_map[current_section_id].get("parent_section")
            if not parent_section_id:
                return current_section_id
            current_section_id = parent_section_id

        return section_id

    def _get_section_title(self, section_id: str) -> str:
        if section_id == "_ungrouped":
            return "Other Generated Text"

        top_level_section_id = self._get_top_level_section_id(section_id)

        for section in self.template_sections or []:
            if section.get("id") == top_level_section_id:
                return section.get("title", top_level_section_id)

        return top_level_section_id

    def _get_result_section_id(
        self, content_id: str, result: TextGenerationResult
    ) -> str:
        if result.section_id:
            return self._get_top_level_section_id(result.section_id)

        for section in self.template_sections or []:
            for content in section.get("contents", []):
                if content.get("content_id") == content_id:
                    return self._get_top_level_section_id(section["id"])

        return "_ungrouped"

    def _group_results_by_section(self) -> Dict[str, Dict[str, TextGenerationResult]]:
        grouped_results: Dict[str, Dict[str, TextGenerationResult]] = {}

        for content_id, result in self.results.items():
            section_id = self._get_result_section_id(content_id, result)
            grouped_results.setdefault(section_id, {})[content_id] = result

        return grouped_results

    def to_html(self) -> str:
        """Generate HTML representation of the summary."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        title_html = f"""
        <h2>Generated Documentation Text: <i style="color: #DE257E">{self.title}</i></h2><hr>
        """
        html_parts.append(title_html)

        if self.show_link:
            from ..api_client import get_api_host, get_api_model

            api_host = get_api_host()
            api_model = get_api_model()
            if api_host and api_model:
                ui_host = api_host.replace("/api/v1/tracking", "").replace("api", "app")
                if not ui_host.endswith("/"):
                    ui_host = f"{ui_host}/"
                link = f"{ui_host}model-inventory/{api_model}"
                results_link_html = f"""
                <h3>
                    Check out the updated documentation on
                    <a href="{link}" target="_blank">ValidMind</a>.
                </h3>
                """
                html_parts.append(results_link_html)

        if self.description:
            html_parts.append(
                f'<div class="result">{md_to_html(self.description)}</div>'
            )

        if self.results:
            grouped_results = self._group_results_by_section()
            section_items = []
            section_titles = []

            ordered_section_ids = [
                section["id"]
                for section in self.template_sections or []
                if section.get("id") in grouped_results
            ]
            for section_id in grouped_results:
                if section_id not in ordered_section_ids:
                    ordered_section_ids.append(section_id)

            for section_id in ordered_section_ids:
                section_results = grouped_results[section_id]
                accordion_items = []
                accordion_titles = []

                for content_id, result in section_results.items():
                    accordion_items.append(result.to_html())
                    accordion_titles.append(f"Text Block: '{content_id}'")

                section_items.append(
                    StatefulHTMLRenderer.render_accordion(
                        accordion_items, accordion_titles
                    )
                )
                section_titles.append(self._get_section_title(section_id))

            html_parts.append(
                StatefulHTMLRenderer.render_accordion(section_items, section_titles)
            )

        return f'<div class="vm-documentation-text-summary">{"".join(html_parts)}</div>'
