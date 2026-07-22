# App

The FastAPI service that exposes the orchestrator and RAG over HTTP, and serves
the built web UI.

- `api.py` — FastAPI routes. Serves the web UI at `/`, the analysis endpoints,
  and the RAG endpoints (`POST /rag/ingest`, `/rag/upload`, `/rag/search`,
  `/rag/answer`, and `GET /rag/documents`). API docs at `/docs`.
- `static/` — the built web UI (produced by `make build-ui`).
- `openwebui_pipe.py`, `openwebui_rag_pipe.py` — optional Open WebUI adapters
  (see below).

The primary interface is the built-in web UI at `http://localhost:8000` after
`make compose-up`. Run the API alone with `make run-api`.

## Open WebUI (optional, not maintained)

> Note: Open WebUI was removed from `docker-compose.yml` to keep the stack fast
> for users who do not need it. It is no longer part of the maintained flow, and
> since we moved to the built-in web UI it may not work correctly with newer
> features. Use it only if you specifically want the Open WebUI chat interface.

If you still want it, start it separately:

```bash
make openwebui-up      # starts Open WebUI at http://localhost:3000
```

Then:

1. Open `http://localhost:3000` and create the admin account (first user).
2. Admin Panel → Functions → Add Function.
3. Paste the contents of `openwebui_pipe.py` (full analysis) or
   `openwebui_rag_pipe.py` (RAG only), save, and enable the toggle.
4. The default `mascan_api_url` points at the API container; adjust the Valve if
   your API runs elsewhere.
5. Open a new chat, select the "MAScan" (or "RAG") model, and ask a question.

Stop it with `make openwebui-down`. Follow logs with `make openwebui-logs`.
