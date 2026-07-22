# Web UI

The MAScan web interface: a React app (Vite) that talks to the MAScan API and
streams the analysis, including a live agent graph.

The built app is served by the API at `http://localhost:8000`, so most users
never run the frontend directly.

## Build for the app

After changing the UI, build it into the API's static directory, then start the
stack:

```bash
make build-ui
make compose-up
```

`build-ui` runs `npm install` and `npm run build`. Without a rebuild, the stack
serves the previously built UI.

## Develop

For fast iteration with hot reload:

```bash
make run-api    # start the API on http://localhost:8000
make dev-ui     # start the Vite dev server (proxies to the API)
```

Open the URL that Vite prints (usually `http://localhost:5173`).

## Structure

```text
frontend/
├── index.html
├── vite.config.js
├── package.json
└── src/            React components and the analysis stream hook
```
