import { useEffect, useRef, useState } from "react";
import FileUpload from "./FileUpload";
import MarkdownContent from "./MarkdownContent";
import { buildProgressSteps } from "../progress.js";
import { withoutFactCheck } from "../markdown.js";

// Center pane: messages, the live run indicator, an inline clarification prompt,
// and the composer.
export default function Conversation({ messages, run, onSend, onResume }) {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState("");
  const busy = run.status === "running";
  const showProgress = busy || run.status === "clarification";
  const progressSteps = buildProgressSteps(run);
  const progressKey = progressSteps.map((step) => `${step.id}:${step.status}:${step.text}`).join("|");
  const scrollRef = useRef(null);
  const taRef = useRef(null);

  // Keep the latest content in view as it streams in.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, run.status, run.finalMarkdown, progressKey]);

  // Grow the input with its content, up to a cap, like Claude's composer.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
  }, [input]);

  function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    onSend(q);
  }

  function submitAnswer() {
    const a = answer.trim();
    if (!a) return;
    setAnswer("");
    onResume(a);
  }

  return (
    <main className="conversation">
      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && run.status === "idle" && (
          <div className="empty-state">
            <h1>Scan a market</h1>
            <p>Ask a question and the PESTEL agents will investigate it. Attach a document to add it to the knowledge base.</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            {m.role === "assistant" ? <MarkdownContent>{withoutFactCheck(m.content)}</MarkdownContent> : m.content}
          </div>
        ))}

        {showProgress && progressSteps.length > 0 && <ProgressPanel steps={progressSteps} />}

        {run.status === "clarification" && run.clarification && (
          <div className="msg msg-assistant clarify">
            <p>{run.clarification.question}</p>
            <div className="clarify-input">
              <input
                value={answer}
                placeholder="Type your answer…"
                onChange={(e) => setAnswer(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submitAnswer()}
                autoFocus
              />
              <button onClick={submitAnswer}>Send</button>
            </div>
          </div>
        )}

        {run.status === "error" && (
          <div className="msg msg-assistant error">Something went wrong: {run.error}</div>
        )}
      </div>

      <div className="composer">
        <FileUpload />
        <textarea
          ref={taRef}
          value={input}
          placeholder="Ask about a market…"
          rows={1}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button className="send" onClick={send} disabled={busy || !input.trim()}>
          Send
        </button>
      </div>
    </main>
  );
}

function ProgressPanel({ steps }) {
  return (
    <div className="msg msg-assistant progress-panel" role="status" aria-live="polite">
      <div className="progress-title">Analysis progress</div>
      <div className="progress-steps">
        {steps.map((step) => (
          <div key={step.id} className={`progress-step progress-${step.status}`}>
            <span className="progress-marker" aria-hidden="true">
              {step.status === "active" ? "" : step.status === "failed" ? "×" : "✓"}
            </span>
            <span>{step.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
