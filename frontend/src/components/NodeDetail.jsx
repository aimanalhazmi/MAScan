import { useState } from "react";
import { LABELS } from "../graph";
import { withoutFactCheck } from "../markdown.js";
import MarkdownContent from "./MarkdownContent";

// Inspector for a clicked graph node: the plan for the planner, the agent's
// report for an agent, the final report, or citation validation.
export default function NodeDetail({ run, nodeId, onClose }) {
  const [tab, setTab] = useState("output");
  if (!nodeId) return null;

  const label = LABELS[nodeId] || nodeId;
  const tabs = ["output", "metrics"];

  const handleTabKeyDown = (event, name) => {
    const index = tabs.indexOf(name);
    let nextIndex;

    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = tabs.length - 1;
    else return;

    event.preventDefault();
    setTab(tabs[nextIndex]);
    event.currentTarget.parentElement.children[nextIndex].focus();
  };

  return (
    <div className="detail">
      <div className="detail-head">
        <span>{label}</span>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      <div className="detail-tabs" role="tablist" aria-label={label + " details"}>
        {tabs.map((name) => (
          <button
            key={name}
            className={"detail-tab" + (tab === name ? " active" : "")}
            id={nodeId + "-" + name + "-tab"}
            type="button"
            role="tab"
            aria-selected={tab === name}
            aria-controls={nodeId + "-" + name + "-panel"}
            tabIndex={tab === name ? 0 : -1}
            onClick={() => setTab(name)}
            onKeyDown={(event) => handleTabKeyDown(event, name)}
          >
            {name === "output" ? "Output" : "Metrics"}
          </button>
        ))}
      </div>
      <div
        id={nodeId + "-output-panel"}
        className="detail-body"
        role="tabpanel"
        aria-labelledby={nodeId + "-output-tab"}
        hidden={tab !== "output"}
      >
        {renderBody(run, nodeId)}
      </div>
      <div
        id={nodeId + "-metrics-panel"}
        className="detail-body"
        role="tabpanel"
        aria-labelledby={nodeId + "-metrics-tab"}
        hidden={tab !== "metrics"}
      >
        <Metrics metric={run.componentMetrics[nodeId]} />
      </div>
    </div>
  );
}

function Metrics({ metric }) {
  if (!metric) return <p className="muted">No metrics recorded.</p>;

  const tokens = metric.token_usage || {};
  const rows = [
    ["Runs", Number(metric.run_count || 0).toLocaleString()],
    [
      "Duration",
      Number(metric.duration_seconds || 0).toLocaleString(undefined, {
        maximumFractionDigits: 2,
      }) + "s",
    ],
    ["Input tokens", Number(tokens.input_tokens || 0).toLocaleString()],
    ["Output tokens", Number(tokens.output_tokens || 0).toLocaleString()],
    ["Total tokens", Number(tokens.total_tokens || 0).toLocaleString()],
  ];

  return (
    <dl className="metric-list">
      {rows.map(([label, value]) => (
        <div key={label} className="metric-row">
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
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
    if (!run.validation?.markdown) return <Empty status={run.nodeStatus.validator} />;
    return <MarkdownContent>{run.validation.markdown}</MarkdownContent>;
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
