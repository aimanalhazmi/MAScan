# MAScan
Market Prediction & Analysis in Multi-Agent Systems

---

## Project Overview

**MAScan** is an automated Multi-Agent System (MAS) designed to scan global markets, predict emerging trends, and provide deep-dive analysis. By leveraging a network of specialized AI agents, MAScan transforms raw data from diverse external sources into actionable intelligence, allowing corporations to adjust their strategic decisions in real-time.

---

## Quick start

### Prerequisites
- Python 3.12
- [`uv`](https://github.com/astral-sh/uv)
- `make`
- An OpenAI API key
- Docker installed and running.

### Setup

```bash
git clone <repo-url>
cd mascan

make install

cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Run the reference agent
make run-economics Q="EU manufacturing outlook"

# Run the tests
make test
```

---

## Repository layout

```tree
src/mascan/
├── core/         Settings, logging, LLM client, exceptions
├── contracts/    Shared Pydantic shapes (AgentReport, ToolResult, ...)
├── agents/       PESTEL agents. One folder per agent.
├── tools/        Tools (external APIs, web search). Common + per-agent.
├── orchestrator/ placeholder.
└── app/           placeholder.
scripts/          Standalone CLIs for development.
tests/            Unit tests.
```

---

## Key concepts

### Three tool-calling modes

Every agent decides how it uses its tools. All three modes share the same
`BaseTool` classes — pick the right one for your agent.

| Mode | What it does | When to use |
|---|---|---|
| **A — Code-driven** | You call `self.tools["x"].run(...)` in code. | Known, fixed workflow. Predictable |
| **B — LLM-driven** | The LLM picks which tools to call via `create_react_agent`. | LLM should choose among many tools dynamically. |
| **C — Mixed** | Always-call tools + LLM-optional tools. | One or two core tools + situational ones. |

Reference implementation: `src/mascan/agents/economics/agent.py` (Mode C).

---

## How to add a new agent

Every agent has the same shape. Copy the Economics agent and adapt.

1. **Create the folder structure:**

```bash
   mkdir -p src/mascan/agents/<your_agent>/tools
   touch src/mascan/agents/<your_agent>/{__init__.py,agent.py,config.yaml}
   touch src/mascan/agents/<your_agent>/tools/__init__.py
```

2. **Write `config.yaml`:**

```yaml
   name: <your_agent>
   model: gpt-4o-mini
   temperature: 0.2
   max_tokens: 2000
   system_prompt: |
     You are the <Your Domain> analyst in a PESTEL multi-agent system.
     Analyze ...
   tools:
     - web_search          # shared tool
     # add agent-specific tool names here once you create them
```

3. **Write `agent.py`:** Copy `src/mascan/agents/economics/agent.py`. Change:
   - Class name → `YourAgentAgent`
   - `name = "your_agent"` (must match `config.yaml`)
   - `ALWAYS_CALL_TOOLS` and `OPTIONAL_TOOLS` constants for your tools
   - The markdown heading in `render_markdown`

4. **(Optional) Add agent-specific tools** under `agents/<your_agent>/tools/`.
   Copy the stub tool from Economics and adapt. See *How to add a new tool* below.

5. **Register your agent in `__init__.py`:**

```python
   from mascan.agents.registry import agent_registry
   from mascan.agents.<your_agent>.agent import YourAgentAgent
   from mascan.tools.registry import tool_registry
   agent_registry.register(YourAgentAgent())
```

6. **Register in `scripts/run_agent.py`:** Add one import line:
```python
   import mascan.agents.<your_agent>
```

7. **(Optional) Add a Makefile shortcut.** Open the `Makefile` and add:
```makefile
   run-<your_agent>:
   	uv run python scripts/run_agent.py <your_agent> "$(Q)"
```
   Now you can do `make run-<your_agent> Q="..."`.

8. **Write a test** under `tests/agents/test_<your_agent>_agent.py`.
   Copy the Economics test as a template.

9. **Try it:**

```bash
   make run-<your_agent> Q="Your test query"
   make test
```

---

## How to add a new tool

Tools live in one of two places:

- **Shared** (`src/mascan/tools/common/`) — usable by any agent. Use this
  ONLY when at least two agents already use the tool.
- **Agent-specific** (`src/mascan/agents/<name>/tools/`) — owned by one agent.

### Recipe (agent-specific case)

1. Create a file: `src/mascan/agents/<name>/tools/<my_tool>.py`.
2. Inherit `BaseTool`, set `name` (unique snake_case) and `description`.
3. (Optional) Define Pydantic `input_schema` and `output_schema` for typed I/O.
4. Implement `run(**kwargs) -> ToolResult`. Use
   `mascan.tools.http_client.http_get` for HTTP calls. Read API keys from
   `mascan.core.settings.get_settings()`.
5. **Never raise on expected errors** — return `ToolResult(success=False, ...)`.
6. Add the tool's name to the agent's `config.yaml` under `tools:`.

Template: `src/mascan/agents/economics/tools/stub_macro_api.py`.

### Tools and modes

- In **Mode A**, the agent calls the tool directly: `self.tools["x"].run(...)`.
- In **Modes B/C**, the same tool is exposed to the LLM via
  `tool.as_langchain_tool()` automatically.

---

## Environment variables

See `.env.example`. Required:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL_DEFAULT` | Default model (`gpt-4o-mini`) |
| `LOG_LEVEL` | Logging verbosity (`INFO`, `DEBUG`, ...) |

Agent-specific API keys (e.g., `FRED_API_KEY`) go in `.env` and should be
read via `mascan.core.settings`. **Document any new env var in `.env.example`.**

---

## Running with FastAPI and Open WebUI

MAScan exposes its orchestrator over HTTP. Open WebUI provides a chat
interface and calls the MAScan API through a small Pipe Function.

### Docker Compose 

**1. Start the stack:**

```bash
make compose-up
```
First run takes ~2 minutes (builds the API image, pulls Open WebUI).
Subsequent runs are instant.

- MAScan API: `http://localhost:8000`
- Open WebUI: `http://localhost:3000`

**2. Create your admin account** at `http://localhost:3000` (first user
becomes admin).

**3. Install the Pipe Function** (one-time):
- Admin Panel → Functions → **+** (Add Function).
- Copy the entire contents of `src/mascan/app/openwebui_pipe.py`.
- Paste, name it `mascan_pestel_analyst`, Save, enable the toggle.
- The Pipe's default `mascan_api_url` is `http://mascan-api:8000` —
  this is correct for compose; no Valve changes needed.

**4. Ask a question** — open a new chat, select "MAScan",
type your query.


## Make commands
 
All day-to-day tasks go through `make`. Run `make help` to see what's available.
 
| Command | What it does                                               |
|---|------------------------------------------------------------|
| `make install` | Create the virtualenv and install dependencies via uv      |
| `make test` | Run pytest                                                 |
| `make lint` | Ruff + mypy checks                                         |
| `make format` | Format and autofix the code                                |
| `make clean` | Remove caches and build artifacts                          |
| `make run-economics Q="..."` | Run the Economics agent on a query                         |
| `make run-<agent> Q="..."` | Same pattern for any agent (see *add a new agent* above)   |
| `make run-api` | Start the MAScan FastAPI server on `http://localhost:8000` |
| `make compose-up` | Start the full stack app (mascan-api + openwebui)          |
| `make compose-down` | Stop and remove the full stack app                         |
| `make compose-logs` | Follow logs from all services                              |
| `make compose-rebuild` | Force a rebuild of the mascan-api image                    |

### If `make` is unavailable
 
Equivalent raw commands (rarely needed):
 
```bash
uv sync --extra dev                                          # install
uv run pytest -v                                             # test
uv run ruff check src tests && uv run mypy src               # lint
uv run python scripts/run_agent.py economics "your query"    # run an agent
uv run uvicorn mascan.app.api:app --host 0.0.0.0 --port 8000 # run the API
```
 
---

## How to contribute

1. **One agent = one branch.** Branch off `main` as `feat/agent-<name>`.
2. **Follow the recipe.** Don't invent new folder layouts. If you think the
   recipe needs to change, raise it as a separate issue first.
3. **Tests required.** Every agent ships with a unit test. Run `make test`
   before pushing.
4. **No orchestrator code yet.** Build
   your agent to be runnable standalone via `make run-<agent>`.
5. **Document env vars.** If your tool needs an API key, add a placeholder
   line to `.env.example` with a comment.
6. **Use the contracts.** Never invent a new return type. Use `AgentReport`,
   `ToolResult`, `Source`,  from `mascan.contracts`.

---

## Tracing with LangSmith

First you will have to register with LangSmith. Make Sure to register an **EU** Account (otherwise you will have to change it in the .env file!).
After that you need to create a project within LangSmith and name it as specified in your .env file (lower and uppercase matters!).
