# https://stackoverflow.com/questions/10858261/how-to-abort-makefile-if-variable-not-set
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

format:
	poetry run black validmind
	poetry run isort validmind

lint:
# don't check max line length for now since black already takes care of it
# and flake8 is too strict where it doesn't need to be
	poetry run flake8 validmind --config .flake8

install:
	poetry install --all-extras
	poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

build:
	poetry build

test:
ifdef ONLY
	# make test ONLY="tests.test_template tests.test_metadata"
	poetry run python -m unittest $(ONLY)
else
	poetry run python -m unittest discover tests
endif

test-integration:
	poetry run python scripts/run_e2e_notebooks.py

docs:
	rm -rf docs/_build
	poetry run pdoc validmind -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/vm-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico -o docs/_build

docs-serve:
	poetry run pdoc validmind -d google -t docs/templates --no-show-source --logo https://vmai.s3.us-west-1.amazonaws.com/vm-logo.svg --favicon https://vmai.s3.us-west-1.amazonaws.com/favicon.ico

version:
	@:$(call check_defined, tag, new semver version tag to use on pyproject.toml)
	@poetry version $(tag)
	@echo "__version__ = \"$$(poetry version -s)\"" > validmind/__version__.py
	@echo "Version updated to $$(poetry version -s)"
	@echo "Commiting changes to pyproject.toml and __version__.py with message: $$(poetry version -s)"
	@git add pyproject.toml validmind/__version__.py
	@git commit -m "$$(poetry version -s)"

generate-test-id-types:
	poetry run python scripts/generate_test_id_type.py

copyright:
	poetry run python scripts/copyright_files.py

verify-copyright:
	poetry run python scripts/verify_copyright.py

verify-exposed-credentials:
	poetry run python scripts/credentials_check.py

ensure-clean-notebooks:
	poetry run python scripts/ensure_clean_notebooks.py

# Quick target to run all checks
check: copyright format lint test verify-copyright verify-exposed-credentials ensure-clean-notebooks

.PHONY: docs
