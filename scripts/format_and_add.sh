#!/bin/bash

make format
poetry run python scripts/copyright_files.py
git add -u
