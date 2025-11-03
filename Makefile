default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


## Install dependencies with uv
install:
	uv sync

## Delete all temporary files
clean:
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf .pytest_cache
	@rm -rf **/.pytest_cache
	@rm -rf __pycache__
	@rm -rf **/__pycache__
	@rm -rf build
	@rm -rf dist

## Run checks (ruff + test)
check:
	uv lock --check
	uv run ruff check .
	uv run ruff format --check . 

## Format files using ruff
format:
	uv run ruff check . --fix
	uv run ruff format .

## Run tests
test:
	uv run pytest --cov=src --cov-report xml --log-level=WARNING --disable-pytest-warnings

## Run api
api:
	uv run uvicorn src.boilerplate.api.main:app --reload

## Build the docker image
docker:
	docker build -f Dockerfile -t podcaster .

