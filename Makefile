.DEFAULT_GOAL := help
.PHONY: help install test lint format clean run-economics run-political run-legal run-social run-environmental run-orchestrator run-orchestrator-docker run-orchestrator-stream run-api dev-ui build-ui openwebui-up openwebui-down openwebui-logs compose-up compose-down compose-logs compose-rebuild
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

run-legal:
	PYTHONPATH=src uv run python scripts/run_agent.py legal "$(Q)"

run-social:
	PYTHONPATH=src uv run python scripts/run_agent.py social "$(Q)"

run-environmental:
	PYTHONPATH=src uv run python scripts/run_agent.py environmental "$(Q)"

run-technological:
	PYTHONPATH=src uv run python scripts/run_agent.py technological "$(Q)"

run-orchestrator:
	PYTHONPATH=src uv run python scripts/run_orchestrator.py "$(Q)"




run-orchestrator-docker:  ## Run the orchestrator inside the running mascan-api container
	docker exec -e PYTHONPATH=/app/src mascan-api python /app/scripts/run_orchestrator.py "$(Q)"

run-orchestrator-stream:
	PYTHONPATH=src uv run python scripts/run_orchestrator.py --stream "$(Q)"

run-api:  ## Run the MAScan FastAPI server on http://localhost:8000
	uv run uvicorn mascan.app.api:app --reload --host 0.0.0.0 --port 8000

dev-ui:  ## Run the web UI dev server (proxies to the API on :8000)
	cd frontend && npm install && npm run dev

build-ui:  ## Build the web UI into the API's static dir (served at :8000)
	cd frontend && npm install && npm run build

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


compose-up:  ## Start the full stack (mascan-api + openwebui) via docker compose
	docker compose up -d --build
	@echo ""
	@echo "App starting:"
	@echo "  - MAScan API:  http://localhost:8000"
	@echo "  - Open WebUI:  http://localhost:3000"
	@echo ""
	@echo "Logs:    make compose-logs"
	@echo "Stop:    make compose-down"

compose-down:  ## Stop the full stack (data is preserved in named volume)
	docker compose down

compose-logs:  ## Follow logs from all services
	docker compose logs -f

compose-rebuild:  ## Force a rebuild of the mascan-api image
	docker compose build --no-cache mascan-api
