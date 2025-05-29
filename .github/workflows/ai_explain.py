import json
import os
import sys

from github import Github
from openai import APIConnectionError, OpenAI, OpenAIError


# Initialize GitHub and OpenAI clients
github_token = os.getenv("GITHUB_TOKEN")
repo_name = os.getenv("GITHUB_REPOSITORY")
pr_ref = os.getenv("GITHUB_REF")

# Extract PR number from the ref (refs/pull/{pr_number}/merge)
pr_number = pr_ref.split("/")[2]

g = Github(github_token)
repo = g.get_repo(repo_name)
pr = repo.get_pull(int(pr_number))

# Get the files changed in the current PR
files = pr.get_files()
diffs = []

for file in files:
    filename = file.filename
    patch = file.patch
    diffs.append(f"File: {filename}\n{patch}")

diff = "\n\n".join(diffs)

# Add a unique marker at the start of the comment to find comments by the bot
COMMENT_MARKER = "<!-- AI-EXPLAIN-COMMENT -->"

# Fetch existing AI explanation comment
existing_explanation_comments = []
comments = pr.get_issue_comments()
for comment in comments:
    if comment.user.login == "github-actions[bot]" and COMMENT_MARKER in comment.body:
        existing_explanation_comments.append(comment)

# OpenAI prompt template
prompt_template = """
You are an expert software engineer reviewing a pull request (PR) that introduces enhancements or
bugfixes to a software project. Your task is to assess the changes made in the PR and provide a
detailed summary, test suggestions, code quality assessment, and security assessment.

To produce an assessment that can be processed with a script, you generate a JSON object that
describes a PR diff. Your response should be a JSON object with the following keys:

- title (string)
- summary (markdown string that starts with the title "# PR Summary")
- test_suggestions (array of strings)
- code_quality_assessment (array of strings)
- security_assessment (array of strings)

## Instructions for Computing the value of each field

- The `title` field should be a concise summary of the changes.
- The `summary` field should provide a detailed markdown description of the PR, titled "# PR Summary".
    - The `summary` should focus on the functional changes introduced by the PR.
    - The `summary` should omit mentioning version updates from files like `pyproject.toml` or `package.json`, formatting changes, or other trivial modifications.
- The `test_suggestions` field should list test suggestions as an array of strings.
- The `code_quality_assessment` field should provide an assessment of the code quality as an array of strings. It can be "None" if there are no specific concerns.
- The `security_assessment` field should provide an assessment of the security implications as an array of strings. It can be "None" if there are no specific concerns.

diff:
```
{diff}
```
"""

# Prepare OpenAI prompt
prompt = prompt_template.format(diff=diff)

try:
    # Call OpenAI API
    client = OpenAI()
    response = client.chat.completions.create(
        model="o3-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
except APIConnectionError as e:
    print(f"OpenAI API connection error: {e}")
    sys.exit(0)  # happy exit so that the workflow will continue
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
    sys.exit(0)  # happy exit so that the workflow will continue
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(0)  # happy exit so that the workflow will continue

# Parse OpenAI response
ai_response = json.loads(response.choices[0].message.content.strip())

# Create a new comment and delete the existing explanation comment
new_comment = pr.create_issue_comment(
    f"{COMMENT_MARKER}\n"
    f"{ai_response['summary']}\n\n"
    f"## Test Suggestions\n"
    f"- " + "\n- ".join(ai_response["test_suggestions"])
    if ai_response.get("test_suggestions", None)
    else (
        "n/a" + "\n\n## Code Quality Assessment\n- " + "\n- ".join(ai_response["code_quality_assessment"])
        if ai_response.get("code_quality_assessment", None)
        else (
            "n/a" + "\n\n## Security Assessment\n- " + "\n- ".join(ai_response["security_assessment"])
            if ai_response.get("security_assessment", None)
            else "n/a" + "\n\n"
        )
    )
)

# Delete all previous AI explain comments
for comment in existing_explanation_comments:
    try:
        comment.delete()
    except Exception as e:
        print(f"Failed to delete comment: {e!s}")
