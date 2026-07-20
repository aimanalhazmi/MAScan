# Orchestrator

This folder contains the LangGraph orchestrator.

Core files:
- `state.py` — GraphState (generalized: `dict[str, AgentReport]`)
- `planner.py` — Planner node (decides which agents to run and tasks)
- `synthesizer.py` — Synthesizer node (merges reports and adds numbered citations)
- `attribution.py` — Markdown AST parsing and claim-citation attribution
- `validator.py` — staged Link Works, Relevant Content, and Fact Check validation
- `adapters.py` — `make_agent_node(agent)` wraps any BaseAgent into a node
- `graph.py` — Builds the StateGraph

The graph fans out from the planner to the selected PESTEL agents, joins at the
synthesizer, resolves every unique web or uploaded-file citation in the final
Summary, and validates each claim-citation pair in stages before returning the report.
