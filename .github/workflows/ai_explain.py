import os
import json

from openai import OpenAI
from github import Github
from tiktoken import encoding_for_model

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

# Fetch existing AI explanation comment
existing_explanation_comment = None
comments = sorted(pr.get_issue_comments(), key=lambda x: x.created_at, reverse=True)
for comment in comments:
    if comment.user.login == "github-actions[bot]":
        existing_explanation_comment = comment
        break

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

# check number of tokens
encoding = encoding_for_model("gpt-4o-mini")
tokens = encoding.encode(prompt)
print(f"Number of tokens: {len(tokens)}")
# 128k is max tokens for gpt-4o-mini
num_output_tokens = 1000
if len(tokens) > 128000 - num_output_tokens:
    tokens = tokens[: 128000 - num_output_tokens]
    prompt = encoding.decode(tokens)

# Call OpenAI API
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    temperature=0,
)

# Parse OpenAI response
ai_response = json.loads(response.choices[0].message.content.strip())

# Create a new comment and delete the existing explanation comment
new_comment = pr.create_issue_comment(
    f"{ai_response['summary']}\n\n"
    f"## Test Suggestions\n"
    f"- " + "\n- ".join(ai_response["test_suggestions"])
    if ai_response.get("test_suggestions", None)
    else (
        "n/a" + "\n\n"
        f"## Code Quality Assessment\n"
        f"- " + "\n- ".join(ai_response["code_quality_assessment"])
        if ai_response.get("code_quality_assessment", None)
        else (
            "n/a" + "\n\n"
            f"## Security Assessment\n"
            f"- " + "\n- ".join(ai_response["security_assessment"])
            if ai_response.get("security_assessment", None)
            else "n/a" + "\n\n"
        )
    )
)
if existing_explanation_comment:
    existing_explanation_comment.delete()
