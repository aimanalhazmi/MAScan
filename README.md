# MAScan


MAScan is a multi-agent system that analyzes markets and predicts trends. A set
of specialised PESTEL agents (Political, Economic, Social, Technological,
Environmental, Legal) gathers data from external sources, and an orchestrator
plans the work, merges the results, and returns one cited report.

## Prerequisites

- [Docker](https://www.docker.com/) installed and running
- [`uv`](https://github.com/astral-sh/uv) and `make` (only for local development and tests)
- [`npm`](https://www.npmjs.com/)
- Python 3.12 (only for local development)
- An OpenAI API key

## Setup

```bash
git clone <repo-url>
cd mascan

# 1. Install dependencies first — this provides the CLI tools (e.g. `rdt`),
#    tests, and local development.
make install            # or: uv sync --extra dev

# 2. Create your env file and fill in your keys (see "Environment variables").
cp .env.example .env

# 3. (Optional) Log in to Reddit once, so the Reddit tool can authenticate.
#    This needs step 1 to be done first.
rdt login
```

## Environment variables

All variables live in `.env` (copied from `.env.example`). The most important:

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Everything | Required. |
| `FIRECRAWL_API_KEY` | Web search | Required for the `web_search` tool. Get a free key at [firecrawl.dev](https://www.firecrawl.dev/). Docker Compose runs a self-hosted Firecrawl, so a key is optional there. |
| `NEWS_API_KEY` | News tool | Optional. Free key at [newsdata.io](https://newsdata.io). |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | X / Twitter tool | Optional. Log in to x.com in a browser, open DevTools → Application → Cookies → `https://x.com`, and copy the `auth_token` and `ct0` cookie values. |
| Reddit | Reddit tool | Optional. Run `rdt login` once on your machine. It caches a session that the tool reuses; Docker Compose mounts that session read-only. |
| `DATABASE_URL` | RAG | Set by Docker Compose. Leave unset to disable RAG. See [`src/mascan/rag/README.md`](src/mascan/rag/README.md). |
| `LANGSMITH_API_KEY` | Tracing | Optional. Use an EU account and match `LANGSMITH_PROJECT`. |
| `SEMANTIC_SCHOLAR_API_KEY` | Scholar Search Tool | Optional. Request a free API key for access |

The full list, with comments, is in `.env.example`.

## How to run

Docker is the supported way to run MAScan.

> Note: running outside Docker (for example `make run-api` locally) may fail,
> because the `web_search` tool uses a Firecrawl instance that is hosted locally
> inside the Docker stack. If you instead use cloud Firecrawl, set
> `FIRECRAWL_API_KEY` (and leave `FIRECRAWL_API_URL` empty) in `.env` and it
> will work without the container.

There are two ways to use MAScan. Both need the stack running.

### 1. Start the stack

```bash
make compose-up
```

The first run takes a few minutes (it builds the API image and starts Postgres
and Firecrawl). It then serves:

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

If you changed the frontend, rebuild it first so the new UI is served:

```bash
make build-ui
make compose-up
```

Stop the stack with `make compose-down`. Follow logs with `make compose-logs`.

### 2a. Use the web UI

Open `http://localhost:8000`, type a market question, and read the report.

### 2b. Use the command line

With the stack running:

```bash
make run-orchestrator Q="Impact of EU carbon rules on German car makers"
make run-economics    Q="EU manufacturing outlook in 2027"
```

`run-orchestrator` runs the full multi-agent analysis. `run-<agent>` runs a
single agent (`economics`, `political`, `legal`, `social`, `environmental`,
`technological`).

## Documentation

| Topic | Where |
|---|---|
| Contributing, adding agents and tools | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Evaluation (gold standard, scenarios) | [`src/mascan/eval/README.md`](src/mascan/eval/README.md) |
| Retrieval (RAG) | [`src/mascan/rag/README.md`](src/mascan/rag/README.md) |
| API and Open WebUI | [`src/mascan/app/README.md`](src/mascan/app/README.md) |
| Web UI (frontend) | [`frontend/README.md`](frontend/README.md) |
| Orchestrator | [`src/mascan/orchestrator/README.md`](src/mascan/orchestrator/README.md) |

## Repository layout

```text
src/mascan/
├── core/          Settings, logging, LLM client
├── contracts/     Shared Pydantic shapes (AgentReport, ToolResult, ...)
├── agents/        PESTEL agents, one folder per agent
├── tools/         Tools (external APIs, web search)
├── rag/           Retrieval-Augmented Generation
├── orchestrator/  Planner, agent fan-out, synthesizer
├── eval/          Evaluation code and its README
├── app/           FastAPI service and web UI
└── utils/         Shared helpers
frontend/          React web UI (built into the API)
scripts/           Standalone CLIs
eval_papers/       Evaluation cases (PDFs, manifests)
tests/             Unit tests
```

## Make commands

Run `make help` for the full list. The common ones:

| Command | What it does |
|---|---|
| `make compose-up` | Start the full stack (API + UI + Postgres + Firecrawl) |
| `make compose-down` | Stop the stack (data is preserved) |
| `make build-ui` | Build the web UI into the API |
| `make run-orchestrator Q="..."` | Run the full multi-agent analysis |
| `make run-<agent> Q="..."` | Run a single agent |
| `make install` | Install dependencies (local development) |
| `make test` | Run the tests |
| `make lint` / `make format` | Static checks / auto-format |
