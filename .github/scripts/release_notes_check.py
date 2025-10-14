import json
import os
import re

import requests
from github import Auth, Github


def get_pull_request_number(pr_url):
    response = requests.get(pr_url, timeout=10)
    if response.status_code == 200:
        pr_number = int(re.search(r"pull/(\d+)", response.url).group(1))
        return pr_number
    else:
        raise ValueError("Failed to fetch pull request details.")


def ci_check(pr_number, access_token):
    auth = Auth.Token(access_token)
    g = Github(auth=auth)
    # Get repository, pull request, and labels
    repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])
    pr = repo.get_pull(pr_number)
    labels = [label.name for label in pr.labels]
    description = pr.body

    # Check for the presence of 'internal' or 'dependencies' label
    if "internal" in labels or "dependencies" in labels:
        return True

    required_labels = [
        "highlight",
        "enhancement",
        "bug",
        "deprecation",
        "documentation",
    ]

    # Check if root comment is empty
    if description is None or not description.strip():
        if not any(label in labels for label in required_labels):
            comment = "Pull requests must include at least one of the required labels: `internal`, `highlight`, `enhancement`, `bug`, `deprecation`, `documentation`. Except for `internal`, pull requests must also include a description in the release notes section."
            pr.create_issue_comment(comment)
            return False
        comment = "Pull requests must include a description in the release notes section."
        pr.create_issue_comment(comment)
        return False

    # Check for the presence of at least one label
    if not any(label in labels for label in required_labels):
        # Check for description of external change
        release_notes_pattern = r"## (?:External )?Release Notes[\n\r]+(.*?)(?:\n##|\Z)"
        release_notes_match = re.search(release_notes_pattern, description, re.DOTALL | re.IGNORECASE)
        if release_notes_match:
            release_notes_text = release_notes_match.group(1).strip()
            if release_notes_text and release_notes_text != "<!--- REPLACE THIS COMMENT WITH YOUR DESCRIPTION --->":
                comment = "Pull requests must include at least one of the required labels: `internal` (no release notes required), `highlight`, `enhancement`, `bug`, `deprecation`, `documentation`. Except for `internal`, pull requests must also include a description in the release notes section."
                pr.create_issue_comment(comment)
                return False
        # Pull request has neither a label nor a description
        comment = "Pull requests must include at least one of the required labels: `internal`, `highlight`, `enhancement`, `bug`, `deprecation`, `documentation`. Except for `internal`, pull requests must also include a description in the release notes section."
        pr.create_issue_comment(comment)
        return False

    # Check for description of external change
    release_notes_pattern = r"## (?:External )?Release Notes[\n\r]+(.*?)(?:\n##|\Z)"
    release_notes_match = re.search(release_notes_pattern, description, re.DOTALL | re.IGNORECASE)
    if release_notes_match:
        release_notes_text = release_notes_match.group(1).strip()
        if release_notes_text and release_notes_text != "<!--- REPLACE THIS COMMENT WITH YOUR DESCRIPTION --->":
            return True

    comment = "Pull requests must include a description in the release notes section."
    pr.create_issue_comment(comment)
    return False


if __name__ == "__main__":
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path) as f:
        event_payload = json.load(f)
        pr_number = event_payload["pull_request"]["number"]

    access_token = os.environ["GITHUB_TOKEN"]

    result = ci_check(pr_number, access_token)

    if not result:
        exit(1)

    exit(0)
