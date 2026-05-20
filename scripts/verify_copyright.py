# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution or use of this software is strictly prohibited.
# Please refer to the LICENSE file in the root directory of this repository
# for more information.
#
# Copyright © 2023 ValidMind Inc. All rights reserved.

"""
This script verifies that all files under a directory have the
ValidMind copyright block as a header.

How to use:
    uv run python scripts/verify_copyright.py
"""

import os

copyright_path = os.path.join(os.getcwd(), "scripts", "copyright.txt")
with open(copyright_path) as f:
    copyright = f.read()

# Scan the Python package directory
directory = os.path.join(os.getcwd(), "validmind")

# List of file extensions to process
extensions = [".py"]

# Invalid files
errors = []

# Loop through all files in the directory and its subdirectories
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file has a valid extension
        if file.endswith(tuple(extensions)) and file != "__version__.py":
            # Read the contents of the file
            with open(os.path.join(root, file), "r") as f:
                contents = f.read()

            # Check if the file is __init__.py and if it's empty
            if file == "__init__.py" and not contents.strip():
                continue

            # Check if the file contains the copyright header
            if not contents.startswith(copyright):
                errors.append(
                    f"File {os.path.join(root, file)} does not have the copyright header."
                )

if errors:
    print("\n".join(errors))
    print("\nPlease fix the errors above by running `make copyright`")
    exit(1)
