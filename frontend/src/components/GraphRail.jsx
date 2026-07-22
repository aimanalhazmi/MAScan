import { useMemo, useState } from "react";
import ReactFlow, { Background } from "reactflow";
import { AGENTS, LABELS } from "../graph";
import { aggregateRunTokenUsage } from "../useAnalysisStream.js";

// Fixed positions for the PESTEL pipeline. Planner on top, the six agents in a
// 2-column grid, followed by synthesizer and validator.
const POS = {
  planner: { x: 95, y: 0 },
  clarify: { x: 260, y: 55 },
  political: { x: 5, y: 95 },
  economics: { x: 175, y: 95 },
  social: { x: 5, y: 175 },
  technological: { x: 175, y: 175 },
  environmental: { x: 5, y: 255 },
  legal: { x: 175, y: 255 },
  synthesizer: { x: 95, y: 350 },
  validator: { x: 95, y: 430 },
};

function statusClass(status, selected) {
  return `scan-node scan-${status}${selected ? " scan-selected" : ""}`;
}

function RunMetrics({ run }) {
  const tokens = aggregateRunTokenUsage(run.componentMetrics);
  const active =
    run.status === "running" || run.status === "clarification";
  const duration = active
    ? "Running…"
    : run.runDurationSeconds == null
      ? "Not recorded"
      : Number(run.runDurationSeconds).toLocaleString(undefined, {
          maximumFractionDigits: 2,
        }) + "s";
  const rows = [
    ["Wall-clock duration", duration],
    ["Input tokens", Number(tokens.input_tokens).toLocaleString()],
    ["Output tokens", Number(tokens.output_tokens).toLocaleString()],
    ["Total tokens", Number(tokens.total_tokens).toLocaleString()],
  ];

  return (
    <div className="run-metrics">
      <dl className="metric-list">
        {rows.map(([label, value]) => (
          <div key={label} className="metric-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function GraphRail({ run, selected, onSelect }) {
  const [view, setView] = useState("scan");

  const showView = (nextView) => {
    setView(nextView);
    if (nextView === "metrics") onSelect(null);
  };

  const nodes = useMemo(() => {
    return Object.keys(POS).map((id) => ({
      id,
      position: POS[id],
      data: { label: LABELS[id] },
      className: statusClass(run.nodeStatus[id], selected === id),
      draggable: false,
      connectable: false,
    }));
  }, [run.nodeStatus, selected]);

  const edges = useMemo(() => {
    const active = (id) =>
      run.nodeStatus[id] === "active" || run.nodeStatus[id] === "done";
    const list = [];
    // Planner branches to the clarification node when it needs more input.
    const clarify = run.nodeStatus.clarify;
    list.push({
      id: "p-clarify",
      source: "planner",
      target: "clarify",
      animated: clarify === "active",
      className: clarify === "idle" ? "scan-edge" : "scan-edge-on",
    });
    for (const a of AGENTS) {
      const on = active("planner") && run.nodeStatus[a] !== "skipped" && run.nodeStatus[a] !== "idle";
      list.push({ id: `p-${a}`, source: "planner", target: a, animated: on, className: on ? "scan-edge-on" : "scan-edge" });
      const on2 = run.nodeStatus[a] === "done";
      list.push({ id: `${a}-s`, source: a, target: "synthesizer", animated: on2, className: on2 ? "scan-edge-on" : "scan-edge" });
    }
    const validate = run.nodeStatus.validator === "active" || run.nodeStatus.validator === "done";
    list.push({
      id: "s-validator",
      source: "synthesizer",
      target: "validator",
      animated: run.nodeStatus.validator === "active",
      className: validate ? "scan-edge-on" : "scan-edge",
    });
    return list;
  }, [run.nodeStatus]);

  return (
    <div className="graph-rail">
      <div className="rail-head" role="group" aria-label="Right rail view">
        <button
          type="button"
          className={"rail-view-button" + (view === "scan" ? " active" : "")}
          aria-pressed={view === "scan"}
          onClick={() => showView("scan")}
        >
          Scan
        </button>
        <button
          type="button"
          className={"rail-view-button" + (view === "metrics" ? " active" : "")}
          aria-pressed={view === "metrics"}
          onClick={() => showView("metrics")}
        >
          Run Metrics
        </button>
      </div>
      {view === "scan" ? (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodeClick={(_, n) => onSelect(n.id)}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          panOnDrag={false}
          zoomOnScroll={false}
          zoomOnDoubleClick={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={18} size={1} className="rail-bg" />
        </ReactFlow>
      ) : (
        <RunMetrics run={run} />
      )}
    </div>
  );
}
