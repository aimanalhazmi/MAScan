# App

A Streamlit interface and may later be augmented by a
FastAPI service. Both consume a single entrypoint exposed by the
orchestrator (`run(user_input) -> FinalReport`), so swapping the frontend
does not touch core code.

Planned:
- `streamlit_app.py` — interactive UI for sending queries and viewing reports
- `api.py` — (later) FastAPI routes