.DEFAULT_GOAL := help
.PHONY: help compose-up compose-down compose-logs compose-rebuild mascan-logs run-orchestrator run-orchestrator-stream run-economics run-political run-legal run-social run-environmental run-technological build-ui dev-ui run-api install test lint format clean gold-eval-pre gold-eval gold-eval-post market-scenario-eval-pre market-scenario-eval openwebui-up openwebui-down openwebui-logs
help:  ## Show this help message
	@echo "MAScan — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage examples:"
	@echo "  make compose-up"
	@echo "  make run-orchestrator Q=\"EU manufacturing outlook in 2027\""
	@echo "  make run-economics Q=\"EU manufacturing outlook in 2027\""


# Stack (Docker)

compose-up:  ## Start the full stack (API + UI + Postgres + Firecrawl)
	docker compose up -d --build
	@echo ""
	@echo "App starting:"
	@echo "  - MAScan UI:  http://localhost:8000"
	@echo "  - MAScan API docs:  http://localhost:8000/docs"
	@echo ""
	@echo "Logs:    make compose-logs"
	@echo "Stop:    make compose-down"

compose-down:  ## Stop the full stack (data is preserved in named volume)
	docker compose down

compose-logs:  ## Follow logs from all services
	docker compose logs -f

mascan-logs:  ## Follow logs from the mascan api container
	docker logs -f api

compose-rebuild:  ## Force a rebuild of the mascan-api image
	docker compose build --no-cache mascan-api


# ── Run analyses (stack must be up) ──────────────────────────────────────────

run-orchestrator:  ## Run the full multi-agent analysis, e.g. Q="..."
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_orchestrator.py "$(Q)"

run-orchestrator-stream:  ## Run the orchestrator with streaming output
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_orchestrator.py --stream "$(Q)"

run-economics:  ## Run one agent, e.g. run-<agent> Q="..." (economics/political/legal/social/environmental/technological)
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py economics "$(Q)"

run-political:
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py political "$(Q)"

run-legal:
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py legal "$(Q)"

run-social:
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py social "$(Q)"

run-environmental:
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py environmental "$(Q)"

run-technological:
	docker exec -it -e PYTHONPATH=/app/src api python /app/scripts/run_agent.py technological "$(Q)"


# Web UI and API

build-ui:  ## Build the web UI into the API's static dir (served at :8000)
	cd frontend && npm install && npm run build

dev-ui:  ## Run the web UI dev server (proxies to the API on :8000)
	cd frontend && npm install && npm run dev

run-api:  ## Run the MAScan FastAPI server locally on http://localhost:8000
	uv run uvicorn mascan.app.api:app --reload --host 0.0.0.0 --port 8000


# Local development

install:  ## Install dependencies into a local virtualenv (uv)
	uv sync --extra dev

test:  ## Run the test suite (pytest)
	uv run pytest -v

lint:  ## Run static checks (ruff + mypy)
	uv run ruff check src tests
	uv run mypy src

format:  ## Auto-format and autofix the code (ruff)
	uv run ruff format src tests
	uv run ruff check --fix src tests

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	find . -type f -name "*.pyc" -delete


# Evaluation

gold-eval-pre:  ## Preview gold-standard eval commands (no API calls)
	PYTHONPATH=src uv run python scripts/run_gold_pre_human.py --manifest eval_papers/gold_experiment_manifest.example.json --reviewer-out-dir eval_results/human_reviewers --trace-csv-out eval_results/case_trace.csv --preflight-out eval_results/pre_human_preflight.json --preflight-markdown-out eval_results/pre_human_preflight.md

gold-eval:  ## Run the paid gold-standard pre-human phase (responses, judge, human packet)
	PYTHONPATH=src uv run python scripts/run_gold_pre_human.py --manifest eval_papers/gold_experiment_manifest.example.json --reviewer-out-dir eval_results/human_reviewers --trace-csv-out eval_results/case_trace.csv --preflight-out eval_results/pre_human_preflight.json --preflight-markdown-out eval_results/pre_human_preflight.md --execute

gold-eval-post:  ## Run post-human phase after raters return CSV files
	PYTHONPATH=src uv run python scripts/run_gold_post_human.py --manifest eval_papers/gold_experiment_manifest.example.json --ratings-csv eval_results/human_reviewers/rater_1_ratings.csv eval_results/human_reviewers/rater_2_ratings.csv eval_results/human_reviewers/rater_3_ratings.csv eval_results/human_reviewers/rater_4_ratings.csv eval_results/human_reviewers/rater_5_ratings.csv --preflight-out eval_results/post_human_preflight.json --preflight-markdown-out eval_results/post_human_preflight.md --execute

market-scenario-eval-pre:  ## Preview 3-case market scenario eval (no API calls)
	PYTHONPATH=src uv run python scripts/run_market_scenario_eval.py

market-scenario-eval:  ## Run paid 3-case market scenario eval (MAScan vs zero-shot)
	PYTHONPATH=src uv run python scripts/run_market_scenario_eval.py --execute --init-pricing


# Open WebUI (optional, not maintained)

openwebui-up:  ## Start Open WebUI in Docker on http://localhost:3000
	@docker ps -a --format '{{.Names}}' | grep -q '^mascan-openwebui$$' && \
		echo "Container already exists. Run 'docker start mascan-openwebui' to resume, or 'make openwebui-down' to remove it first." && exit 1 || true
	docker run -d \
		--name mascan-openwebui \
		--add-host=host.docker.internal:host-gateway \
		-p 3000:8080 \
		-v mascan-openwebui-data:/app/backend/data \
		--restart unless-stopped \
		ghcr.io/open-webui/open-webui:main
	@echo ""
	@echo "Open WebUI starting at http://localhost:3000 (wait ~15s)"

openwebui-down:  ## Stop and remove the Open WebUI container (data is preserved)
	-docker stop mascan-openwebui
	-docker rm mascan-openwebui

openwebui-logs:  ## Follow Open WebUI container logs
	docker logs -f mascan-openwebui
