	# https://stackoverflow.com/questions/10858261/how-to-abort-makefile-if-variable-not-set
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

format:
	uv run black validmind
	uv run isort validmind

lint:
# don't check max line length for now since black already takes care of it
# and flake8 is too strict where it doesn't need to be
	uv run flake8 validmind --config .flake8

install:
	uv sync --all-extras --group dev
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

build:
	uv build

test:
ifdef ONLY
	# make test ONLY="tests.test_template tests.test_metadata"
	uv run python -m unittest $(ONLY)
else
	uv run python -m unittest discover tests
endif

test-unit:
	uv run python -m unittest "tests.test_unit_tests"

test-integration:
	uv run python scripts/run_e2e_notebooks.py

docs:
	rm -rf docs/_build
ifeq ($(shell uname),Darwin)
	uv run pdoc validmind !validmind.tests.data_validation.Protected* -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/validmind-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico -o docs/_build
else
	uv run pdoc validmind -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/validmind-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico -o docs/_build
endif

docs-serve:
ifeq ($(shell uname),Darwin)
	uv run pdoc validmind !validmind.tests.data_validation.Protected* -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/validmind-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico
else
	uv run pdoc validmind -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/validmind-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico
endif

quarto-docs:
	# Clean old files
	rm -f docs/validmind.json
	rm -rf docs/validmind
	mkdir -p docs/validmind

	# Generate API JSON dump
	uv run python -m griffe dump validmind -f -o docs/validmind.json -d google -r -U

	# Generate Quarto docs from templates
	uv run python scripts/generate_quarto_docs.py

version:
	@:$(call check_defined, tag, new semver version tag to use on pyproject.toml)
	@if echo "$(tag)" | grep -Eq '^(patch|minor|major)$$'; then uv version --bump $(tag); else uv version "$(tag)"; fi
	@echo "__version__ = \"$$(uv version --short)\"" > validmind/__version__.py
	@sed -i '' 's/^Version: .*/Version: '"$$(uv version --short)"'/' r/validmind/DESCRIPTION
	@echo "Version updated to $$(uv version --short)"
	@echo "Commiting changes to pyproject.toml, __version__.py and r/validmind/DESCRIPTION with message: $$(uv version --short)"
	@git add pyproject.toml validmind/__version__.py r/validmind/DESCRIPTION
	@git commit -m "$$(uv version --short)"

generate-test-id-types:
	uv run python scripts/generate_test_id_type.py

copyright:
	uv run python scripts/copyright_files.py
	uv run python scripts/copyright_notebooks.py

verify-copyright:
	uv run python scripts/verify_copyright.py
	uv run python scripts/verify_notebook_copyright.py

verify-exposed-credentials:
	uv run python scripts/credentials_check.py

ensure-clean-notebooks:
	uv run python scripts/ensure_clean_notebooks.py

# Quick target to run all checks
check: copyright format lint test verify-copyright verify-exposed-credentials ensure-clean-notebooks

.PHONY: docs quarto-docs

notebook:
	uv run python notebooks/templates/e2e_template.py
	git status | grep -v 'notebooks/templates'
