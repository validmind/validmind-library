# Copyright © 2023 ValidMind Inc. All rights reserved.

"""
This script adds a standard ValidMind copyright
block to all Python files in the package directory.

How to use:
    uv run python scripts/copyright_files.py
"""

import os

import nbformat

copyright_path = os.path.join(os.getcwd(), "scripts", "copyright.txt")
with open(copyright_path) as f:
    copyright = f.read()

# Scan the Python package directory
package_directory = os.path.join(os.getcwd(), "validmind")

# Scan the notebooks directory
notebooks_directory = os.path.join(os.getcwd(), "notebooks")

# List of file extensions to process
extensions = [".py", ".ipynb"]


def copyright_python_file(root, file):
    # Check if the file is __init__.py and if it's empty
    if file == "__init__.py":
        with open(os.path.join(root, file), "r") as f:
            contents = f.read()
        if not contents.strip():
            return

    # Read the contents of the file
    with open(os.path.join(root, file), "r") as f:
        lines = f.readlines()

    # Identify the start and end lines
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Copyright"):
            start_index = i
        elif line.strip().startswith("# SPDX-License-Identifier:"):
            end_index = i
            if "AGPL-3.0 AND ValidMind Commercial" not in line:
                file_path = os.path.join(root, file)
                raise Exception(f"Unexpected SPDX identifier in {file_path}. Remove the copyright header manually and rerun the command.")
            break

    # Check for additional comment lines after end_index
    if end_index is not None and end_index + 1 < len(lines) and lines[end_index + 1].strip().startswith("#"):
        file_path = os.path.join(root, file)
        print(f"Warning: Additional lines found after copyright header. Check {file_path}, remove the extra lines if necessary, and rerun the command.")

    # Replace the specified lines if both start and end are found
    if start_index is not None and end_index is not None:
        # Remove lines from start_index to end_index
        del lines[start_index:end_index + 1]
        # Insert the new copyright at start_index
        lines.insert(start_index, copyright)

        # Write the modified contents back to the file
        with open(os.path.join(root, file), "w") as f:
            f.writelines(lines)
    elif start_index is None and end_index is None:
        # Copyright header not found; add it
        contents = "".join(lines)
        contents = f"{copyright}\n{contents}"
        with open(os.path.join(root, file), "w") as f:
            f.write(contents)
    else:
        # Error condition when one of the indices is None but not the other
        file_path = os.path.join(root, file)
        raise Exception(f"Non-standard copyright header found in {file_path}. Remove the header manually and rerun the command.")


def copyright_notebook_file(root, file):
    # Read the contents of the file
    with open(os.path.join(root, file), "r") as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Just override the copyright key on the metadata
    nb.metadata["copyright"] = copyright.strip()

    with open(os.path.join(root, file), "w") as f:
        nbformat.write(nb, f)


# Loop through all files in the directory and its subdirectories
for root, dirs, files in os.walk(package_directory):
    for file in files:
        # Check if the file has a valid extension
        if file.endswith(".py") and file != "__version__.py":
            copyright_python_file(root, file)

# Discuss notebook organization for including 3rd party notebooks
# before adding this change.
#
# for root, dirs, files in os.walk(notebooks_directory):
#     for file in files:
#         if file.endswith(".ipynb"):
#             copyright_notebook_file(root, file)
