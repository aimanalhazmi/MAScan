import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxies backend paths to the FastAPI server so the browser stays
// same-origin (no CORS). Prod builds into the API's static dir, also same-origin.
const backend = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  base: "",
  build: { outDir: "../src/mascan/app/static", emptyOutDir: true },
  server: {
    proxy: {
      "/analyze": backend,
      "/rag": backend,
      "/health": backend,
    },
  },
});
