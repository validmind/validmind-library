"""
Script that extracts test descriptions from test files and generates markdown files with the descriptions.

Usage:
    python scripts/extract_descriptions.py <path>

 - path: path to a test file or directory containing test files
"""

import os
import shutil

import click


output_dir_path = "build/_test_descriptions"


def is_test_function_signature(line, previous_line):
    """
    Test functions should have a @tags or @tasks decorator or a name that starts
    with an uppercase letter.
    """
    return line.startswith("def") and (
        "@tags" in previous_line
        or "@tasks" in previous_line
        or (line.replace("def ", "").strip()[0].isupper())
    )


def retrieve_test_description(path):
    """Generate a test description using gpt4
    You can switch to gpt3.5 if you don't have access but gpt4 should do a better job
    """
    with open(path, "r") as f:
        file_contents = f.read()

    # the description should be inserted after the class definition line
    existing_description_lines = []
    lines = file_contents.split("\n")

    advance_to_next_line = False
    for i, line in enumerate(lines):
        previous_line = lines[i - 1] if i > 0 else ""

        if advance_to_next_line or is_test_function_signature(line, previous_line):
            # ensure this is not a multi-line function signature like this:
            #
            # def test_function(
            #     arg1,
            #     arg2
            # ):
            #
            # we want to keep iterating until we find the closing parenthesis
            if ")" not in line:
                advance_to_next_line = True
                continue

            # check if there is already a doc string for the class
            if '"""' in lines[i + 1]:
                existing_description_lines.append(i + 1)
                j = i + 2
                while j < len(lines):
                    existing_description_lines.append(j)
                    if '"""' in lines[j]:
                        break
                    j += 1

            advance_to_next_line = False
            break

    existing_description_lines = [
        lines[i].strip().strip('"').strip("'") for i in existing_description_lines
    ]
    existing_description_lines = "\n".join(existing_description_lines).strip()

    test_title = path.split("/")[-1].split(".")[0]
    existing_description_lines = f"# {test_title}\n\n{existing_description_lines}"

    return existing_description_lines


def _is_test_file(path):
    return path.endswith(".py") and path.split("/")[-1][0].isupper()


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def main(path):
    """Recursively processes the specified DIRECTORY and updates files needing metadata injection."""
    tests_to_process = []

    # check if path is a file or directory
    if os.path.isfile(path):
        if _is_test_file(path):
            tests_to_process.append(path)
        else:
            raise ValueError(f"File {path} is not a test file")

    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if _is_test_file(file):
                    tests_to_process.append(os.path.join(root, file))

    test_descriptions = []
    for file in tests_to_process:
        description = retrieve_test_description(file)
        test_descriptions.append({"path": file, "description": description})

    click.echo(
        f"Writing {len(test_descriptions)} descriptions to files at build/_test_descriptions/"
    )

    shutil.rmtree(output_dir_path, ignore_errors=True)
    os.makedirs(output_dir_path, exist_ok=True)

    # write markdown files to build/_test_descriptions
    for test_description in test_descriptions:
        path = test_description["path"]
        path = os.path.splitext(path)[0] + ".md"
        description = test_description["description"]

        full_path = os.path.join(output_dir_path, path)
        dir_path = os.path.dirname(full_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(description)


if __name__ == "__main__":
    main()
