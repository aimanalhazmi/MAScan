.PHONY: install test lint format clean run-economics run-political

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
	uv run python scripts/run_agent.py economics "$(Q)"

run-political:
	uv run python scripts/run_agent.py political "$(Q)"