# App

A Streamlit interface and may later be augmented by a
FastAPI service. Both consume a single entrypoint exposed by the
orchestrator (`run(user_input) -> FinalReport`), so swapping the frontend
does not touch core code.

Planned:
- `streamlit_app.py` — interactive UI for sending queries and viewing reports
- `api.py` — (later) FastAPI routes

## Open WebUI pipes

Paste each file under Admin → Functions and enable it.

- `openwebui_pipe.py` — model "MAScan". Runs the full analyze flow. Attached
  files are ingested into our RAG store first.
- `openwebui_rag_pipe.py` — model "RAG". RAG only: attach files to
  ingest, or ask a question to get a grounded answer with sources.