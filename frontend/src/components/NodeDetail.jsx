import { LABELS } from "../graph";
import { withoutFactCheck } from "../markdown.js";
import MarkdownContent from "./MarkdownContent";

// Inspector for a clicked graph node: the plan for the planner, the agent's
// report for an agent, the final report, or the validator's Fact Check.
export default function NodeDetail({ run, nodeId, onClose }) {
  if (!nodeId) return null;

  return (
    <div className="detail">
      <div className="detail-head">
        <span>{LABELS[nodeId] || nodeId}</span>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      <div className="detail-body">{renderBody(run, nodeId)}</div>
    </div>
  );
}

function renderBody(run, nodeId) {
  if (run.failures[nodeId]) {
    return <p className="detail-fail">{run.failures[nodeId]}</p>;
  }

  if (nodeId === "clarify") {
    const log = run.clarifications || [];
    if (!log.length) {
      const waiting = run.nodeStatus.clarify === "active";
      return <p className="muted">{waiting ? "Waiting for your answer…" : "No clarification was needed."}</p>;
    }
    return log.map((item, i) => (
      <div key={i} className="plan-item">
        <h4>Question</h4>
        <p className="muted">{item.question}</p>
        <h4>Answer</h4>
        <p>{item.answer}</p>
      </div>
    ));
  }

  if (nodeId === "planner") {
    const entries = Object.entries(run.plan);
    if (!entries.length) return <Empty status={run.nodeStatus.planner} />;
    return entries.map(([name, a]) => (
      <div key={name} className="plan-item">
        <h4>{LABELS[name] || name}</h4>
        {a.objective_context && <p className="muted">{a.objective_context}</p>}
        <ul>{(a.tasks || []).map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
    ));
  }

  if (nodeId === "synthesizer") {
    if (!run.finalMarkdown) return <Empty status={run.nodeStatus.synthesizer} />;
    return <MarkdownContent>{withoutFactCheck(run.finalMarkdown)}</MarkdownContent>;
  }

  if (nodeId === "validator") {
    if (!run.validationMarkdown) return <Empty status={run.nodeStatus.validator} />;
    return <MarkdownContent>{run.validationMarkdown}</MarkdownContent>;
  }

  const report = run.reports[nodeId];
  if (!report) return <Empty status={run.nodeStatus[nodeId]} />;
  return <MarkdownContent>{report.rendered_markdown || report.findings || ""}</MarkdownContent>;
}

function Empty({ status }) {
  const msg =
    status === "active" ? "Working…"
    : status === "skipped" ? "Not selected for this query."
    : status === "idle" ? "Waiting."
    : "No output.";
  return <p className="muted">{msg}</p>;
}
