"""Tests for documentation text summary rendering."""

from unittest import mock

from validmind.vm_models.text_generation_summary import DocumentationTextSummary


def test_documentation_text_summary_renders_link_and_accordion():
    monitoring_result = mock.Mock()
    monitoring_result.section_id = "monitoring_details"
    monitoring_result.to_html.return_value = "<div>Monitoring body</div>"

    governance_result = mock.Mock()
    governance_result.section_id = None
    governance_result.to_html.return_value = "<div>Governance body</div>"

    summary = DocumentationTextSummary(
        title="Binary classification",
        description="Template for binary classification models.",
        results={
            "monitoring_plan": monitoring_result,
            "governance_plan": governance_result,
        },
        template_sections=[
            {
                "id": "monitoring",
                "title": "Monitoring Framework",
                "contents": [],
            },
            {
                "id": "monitoring_details",
                "title": "Monitoring Details",
                "parent_section": "monitoring",
                "contents": [],
            },
            {
                "id": "governance",
                "title": "Governance and Controls",
                "contents": [{"content_id": "governance_plan"}],
            },
        ],
    )

    with mock.patch(
        "validmind.api_client.get_api_host",
        return_value="https://api.example.com/api/v1/tracking",
    ), mock.patch("validmind.api_client.get_api_model", return_value="Cmd123"):
        html = summary.to_html()

    assert "Generated Documentation Text" in html
    assert "Binary classification" in html
    assert "Check out the updated documentation on" in html
    assert "https://app.example.com/model-inventory/Cmd123" in html
    assert "Monitoring Framework" in html
    assert "Governance and Controls" in html
    assert "Monitoring Details" not in html
    assert ">monitoring<" not in html
    assert ">governance<" not in html
    assert "Text Block: 'monitoring_plan'" in html
    assert "Text Block: 'governance_plan'" in html
    assert "Monitoring body" in html
    assert "Governance body" in html
