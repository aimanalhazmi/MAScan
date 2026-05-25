.DEFAULT_GOAL := help
.PHONY: help install test lint format clean run-economics run-political

help:  ## Show this help message
	@echo "MAScan — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage examples:"
	@echo "  make install"
	@echo "  make run-economics Q=\"EU manufacturing outlook\""
	@echo "  make test"


install:
	uv sync --extra dev

test:
	uv run pytest -v

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	find . -type f -name "*.pyc" -delete

run-economics:
	PYTHONPATH=src uv run python scripts/run_agent.py economics "$(Q)"

run-political:
	PYTHONPATH=src uv run python scripts/run_agent.py political "$(Q)"
