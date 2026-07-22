# Contributing

## Development setup

```bash
make install          # create the virtualenv and install dependencies
make test             # run the tests
make lint             # ruff + mypy
make format           # auto-format and autofix
```

## Rules

1. One agent per branch. Branch off `develop` as `feat/agent-<name>`.
2. Follow the recipes below. Do not invent new folder layouts.
3. Every agent ships with a unit test. Run `make test` before pushing.
4. Document any new environment variable in `.env.example`.
5. Use the shared contracts (`AgentReport`, `ToolResult`, `Source`) from
   `mascan.contracts`. Never invent a new return type.

## Add a new agent

Copy an existing agent (for example Economics) and adapt it.

1. Create the folder structure:

   ```bash
   mkdir -p src/mascan/agents/<your_agent>/tools
   touch src/mascan/agents/<your_agent>/{__init__.py,agent.py,config.yaml}
   touch src/mascan/agents/<your_agent>/tools/__init__.py
   ```

2. Write `config.yaml`:

   ```yaml
   name: <your_agent>
   model: gpt-4o-mini
   temperature: 0.2
   max_tokens: 2000
   system_prompt: |
     You are the <Your Domain> analyst in a PESTEL multi-agent system.
     Analyze ...
   optional_tools:
     - web_search
     # add agent-specific tool names here once you create them
   ```

3. Write `agent.py` by copying an existing agent. Change the class name, set
   `name = "<your_agent>"` (must match `config.yaml`), and update the markdown
   heading in `render_markdown`.

4. (Optional) Add agent-specific tools under `agents/<your_agent>/tools/`.

5. Register the agent in its `__init__.py`:

   ```python
   from mascan.agents.registry import agent_registry
   from mascan.agents.<your_agent>.agent import YourAgentAgent

   agent_registry.register(YourAgentAgent())
   ```

6. Add one import line in `scripts/run_agent.py`:

   ```python
   import mascan.agents.<your_agent>  # noqa: F401
   ```

7. (Optional) Add a Makefile shortcut so you can run `make run-<your_agent>`.

8. Write a test under `tests/agents/test_<your_agent>_agent.py`.

## Add a new tool

Tools live in one of two places:

- Shared (`src/mascan/tools/common/`) — use only when at least two agents need it.
- Agent-specific (`src/mascan/agents/<name>/tools/`) — owned by one agent.

Recipe:

1. Create `src/mascan/agents/<name>/tools/<my_tool>.py`.
2. Inherit `BaseTool`; set `name` (unique, snake_case) and `description`.
3. (Optional) Define a Pydantic `input_schema` for typed arguments.
4. Implement `run(**kwargs) -> ToolResult`. Use
   `mascan.tools.http_client.http_get` for HTTP calls and read keys from
   `mascan.core.settings.get_settings()`.
5. Never raise on expected errors — return `ToolResult(success=False, ...)`.
6. List the tool's name in the agent's `config.yaml`.

Every LLM-driven tool call is logged (arguments and success or failure) by
`BaseTool`, so you do not need to add logging inside each tool.

## Tracing (LangSmith)

Register a LangSmith account (use an EU account, or change
`LANGSMITH_ENDPOINT` in `.env`). Create a project named exactly as
`LANGSMITH_PROJECT` in your `.env`, then set `LANGSMITH_API_KEY`.
