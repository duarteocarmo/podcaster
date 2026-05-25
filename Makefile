default: help

.PHONY: prep
prep: install clean format check

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: install
install: # Install dependencies with uv
	uv sync

.PHONY: clean
clean: # Delete all temporary files
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf .pytest_cache
	@rm -rf **/.pytest_cache
	@rm -rf __pycache__
	@rm -rf **/__pycache__
	@rm -rf build
	@rm -rf dist

.PHONY: check
check: # Run checks with ruff and ty
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

.PHONY: format
format: # Format files using ruff
	uv run ruff check . --fix
	uv run ruff format .
