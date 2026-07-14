import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "reactflow/dist/style.css";
import "./styles.css";

// Show the error instead of a blank white page if a render throws.
class ErrorBoundary extends React.Component {
  state = { error: null };
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, fontFamily: "monospace", color: "#e5484d" }}>
          <h2>Something crashed in the UI</h2>
          <pre style={{ whiteSpace: "pre-wrap" }}>{String(this.state.error?.stack || this.state.error)}</pre>
          <button onClick={() => location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Surface async / event-handler errors too (an error boundary can't catch those).
function showError(msg) {
  let bar = document.getElementById("global-error");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "global-error";
    bar.style.cssText =
      "position:fixed;bottom:0;left:0;right:0;z-index:9999;background:#e5484d;color:#fff;" +
      "font:12px/1.5 monospace;padding:8px 12px;white-space:pre-wrap;max-height:40vh;overflow:auto";
    document.body.appendChild(bar);
  }
  bar.textContent = "UI error: " + msg;
}
window.addEventListener("error", (e) => showError(e.message));
window.addEventListener("unhandledrejection", (e) => showError(String(e.reason)));

console.log("MAScan UI build: interrupt-fixes-1");

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
