import { useState } from "react";
import { AGENTS } from "./graph";

// Drives an analysis run: POSTs to the SSE endpoints, parses the stream, and
// reduces node events into a single run object the UI renders.

export function emptyRun() {
  const nodeStatus = { planner: "idle", clarify: "idle", synthesizer: "idle" };
  for (const a of AGENTS) nodeStatus[a] = "idle";
  return {
    status: "idle", // idle | running | clarification | done | error
    nodeStatus,
    plan: {},
    reports: {},
    failures: {},
    finalMarkdown: "",
    summary: "",
    clarification: null,
    clarifications: [], // answered question/answer pairs, kept for the graph
    error: "",
  };
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

export function reduce(prev, ev) {
  const run = { ...prev, nodeStatus: { ...prev.nodeStatus } };

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
    run.finalMarkdown = update.final_markdown || run.finalMarkdown;
    run.summary = update.final_summary || run.summary;
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
  // Fill any missing fields so a lightweight snapshot still forms a valid run.
  const hydrate = (partial) => (partial ? { ...emptyRun(), ...partial } : emptyRun());
  const [run, setRun] = useState(() => hydrate(initialRun));

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
    reset: (initial) => setRun(hydrate(initial)),
  };
}
