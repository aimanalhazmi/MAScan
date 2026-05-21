# Orchestrator

This folder will hold the LangGraph orchestrator. Owned by the architect.

Planned files:
- `state.py` — GraphState (generalized: `dict[str, AgentReport]`)
- `planner.py` — Planner node (decides which agents to run and tasks)
- `synthesizer.py` — Synthesizer node (merges reports into final summary)
- `adapters.py` — `make_agent_node(agent)` wraps any BaseAgent into a node
- `graph.py` — Builds the StateGraph

Teammates: do NOT add files here. Build your agents independently
against `BaseAgent`. Integration happens automatically via the registry.