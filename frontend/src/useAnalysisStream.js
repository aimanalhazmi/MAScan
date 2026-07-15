import { useState } from "react";
import { AGENTS } from "./graph.js";
import { withoutFactCheck } from "./markdown.js";

// Drives an analysis run: POSTs to the SSE endpoints, parses the stream, and
// reduces node events into a single run object the UI renders.

export function emptyRun() {
  const nodeStatus = {
    planner: "idle",
    clarify: "idle",
    synthesizer: "idle",
    validator: "idle",
  };
  for (const a of AGENTS) nodeStatus[a] = "idle";
  return {
    status: "idle", // idle | running | clarification | done | error
    nodeStatus,
    plan: {},
    reports: {},
    failures: {},
    componentMetrics: {},
    finalMarkdown: "",
    summary: "",
    validationStatus: "",
    validationIssues: [],
    validationMarkdown: "",
    validationPayload: {},
    clarification: null,
    clarifications: [], // answered question/answer pairs, kept for the graph
    error: "",
  };
}

export function hydrateRun(partial) {
  const base = emptyRun();
  if (!partial) return base;
  const hydrated = {
    ...base,
    ...partial,
    nodeStatus: { ...base.nodeStatus, ...(partial.nodeStatus || {}) },
  };
  hydrated.finalMarkdown = withoutFactCheck(hydrated.finalMarkdown);
  return hydrated;
}

function markSynthesizer(run) {
  const planned = Object.keys(run.plan);
  const settled = planned.every(
    (a) => run.nodeStatus[a] === "done" || run.nodeStatus[a] === "failed"
  );
  if (settled && run.nodeStatus.synthesizer === "idle") {
    run.nodeStatus = { ...run.nodeStatus, synthesizer: "active" };
  }
  return run;
}

function mergeComponentMetrics(current, incoming) {
  const merged = { ...current };
  for (const [name, metric] of Object.entries(incoming || {})) {
    const previous = merged[name];
    if (!previous) {
      merged[name] = metric;
      continue;
    }
    const left = previous.token_usage || {};
    const right = metric.token_usage || {};
    merged[name] = {
      ...previous,
      ...metric,
      run_count: (previous.run_count || 0) + (metric.run_count || 0),
      duration_seconds:
        (previous.duration_seconds || 0) + (metric.duration_seconds || 0),
      token_usage: {
        input_tokens: (left.input_tokens || 0) + (right.input_tokens || 0),
        output_tokens: (left.output_tokens || 0) + (right.output_tokens || 0),
        total_tokens: (left.total_tokens || 0) + (right.total_tokens || 0),
      },
    };
  }
  return merged;
}

export function reduce(prev, ev) {
  const run = {
    ...prev,
    nodeStatus: { ...prev.nodeStatus },
    componentMetrics: { ...(prev.componentMetrics || {}) },
  };

  if (ev.event === "start") {
    run.status = "running";
    run.nodeStatus.planner = "active";
    return run;
  }

  if (ev.event === "clarification") {
    run.status = "clarification";
    run.clarification = { question: ev.question, thread_id: ev.thread_id };
    run.nodeStatus.clarify = "active";
    return run;
  }

  if (ev.event === "done") {
    run.status = "done";
    return run;
  }

  if (ev.event === "error") {
    run.status = "error";
    run.error = ev.message || "Unknown error";
    return run;
  }

  // The stream closed. If we never got a "done" event (e.g. a node crashed
  // server-side), don't hang in "running": show the report if it arrived,
  // otherwise surface an error.
  if (ev.event === "eof") {
    if (run.status === "running") {
      if (run.finalMarkdown) run.status = "done";
      else {
        run.status = "error";
        run.error = "The run ended before producing a report. Check the API logs.";
      }
    }
    return run;
  }

  if (ev.event !== "node") return run;

  // A skipped node streams `update: null`, so coalesce (a `= {}` default only
  // covers undefined).
  const { node } = ev;
  const update = ev.update || {};

  if (update.component_metrics && typeof update.component_metrics === "object") {
    run.componentMetrics = mergeComponentMetrics(
      run.componentMetrics,
      update.component_metrics
    );
  }

  if (node === "planner") {
    // Planner is asking for clarification; stay active until the answer comes.
    if (update.info_request) {
      run.nodeStatus.planner = "active";
      return run;
    }
    run.nodeStatus.planner = "done";
    if (update.plan && typeof update.plan === "object") {
      run.plan = update.plan;
      for (const a of AGENTS) {
        run.nodeStatus[a] = a in update.plan ? "active" : "skipped";
      }
    }
    return markSynthesizer(run);
  }

  if (node === "synthesizer") {
    run.nodeStatus.synthesizer = "done";
    run.nodeStatus.validator = "active";
    run.finalMarkdown = withoutFactCheck(update.final_markdown || run.finalMarkdown);
    run.summary = update.final_summary || run.summary;
    return run;
  }

  if (node === "validator") {
    run.nodeStatus.validator = "done";
    run.validationStatus = update.validation_status || run.validationStatus;
    run.validationIssues = update.validation_issues || run.validationIssues;
    run.validationMarkdown = update.validation_markdown || run.validationMarkdown;
    run.validationPayload = update.validation_payload || run.validationPayload;
    return run;
  }

  if (AGENTS.includes(node)) {
    if (update.reports && update.reports[node]) {
      run.reports = { ...run.reports, [node]: update.reports[node] };
      run.nodeStatus[node] = "done";
    } else if (update.failures && update.failures[node]) {
      run.failures = { ...run.failures, [node]: update.failures[node] };
      run.nodeStatus[node] = "failed";
    }
    return markSynthesizer(run);
  }

  return run;
}

// Read an SSE body line by line and apply each JSON event.
async function consume(response, apply) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        apply(JSON.parse(line.slice(5).trim()));
      } catch {
        // Ignore malformed SSE lines rather than killing the stream.
      }
    }
  }
}

export function useAnalysisStream(initialRun) {
  // Fill missing fields, including nodes absent from older saved snapshots.
  const [run, setRun] = useState(() => hydrateRun(initialRun));

  const apply = (ev) => setRun((r) => reduce(r, ev));

  async function post(url, body) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        apply({ event: "error", message: `HTTP ${res.status}` });
        return;
      }
      await consume(res, apply);
      apply({ event: "eof" });
    } catch (err) {
      apply({ event: "error", message: String(err) });
    }
  }

  return {
    run,
    setRun,
    start: (query) => {
      setRun({ ...emptyRun(), status: "running", nodeStatus: { ...emptyRun().nodeStatus, planner: "active" } });
      return post("/analyze/stream", { query });
    },
    // Branch 2 wires /analyze/resume; the reducer already handles the events.
    resume: (thread_id, answer) => {
      setRun((r) => ({
        ...r,
        status: "running",
        clarification: null,
        nodeStatus: { ...r.nodeStatus, clarify: "done" },
        clarifications: [...(r.clarifications || []), { question: r.clarification?.question, answer }],
      }));
      return post("/analyze/resume", { thread_id, answer });
    },
    reset: (initial) => setRun(hydrateRun(initial)),
  };
}
