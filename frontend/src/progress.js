import { AGENTS, LABELS } from "./graph.js";

const AGENT_SYMBOL = {
  active: "●",
  done: "✓",
  failed: "×",
};

// Turn the existing graph state into a compact, deterministic progress list.
// Agent work is deliberately kept on one line so a full PESTEL run does not
// add six near-identical messages to the conversation.
export function buildProgressSteps(run) {
  const nodeStatus = run?.nodeStatus || {};
  const steps = [];

  if (nodeStatus.planner !== "idle" && nodeStatus.planner !== "skipped") {
    if (run.status === "clarification") {
      steps.push({
        id: "planner",
        status: "active",
        text: "Planner needs more information to complete the analysis plan.",
      });
    } else if (nodeStatus.planner === "done") {
      steps.push({
        id: "planner",
        status: "done",
        text: "Planner completed the analysis plan.",
      });
    } else {
      steps.push({
        id: "planner",
        status: "active",
        text: "Planner is preparing the analysis plan…",
      });
    }
  }

  const selectedAgents = AGENTS.filter((agent) =>
    Object.prototype.hasOwnProperty.call(run?.plan || {}, agent)
  );
  if (selectedAgents.length > 0 && nodeStatus.planner === "done") {
    const allSettled = selectedAgents.every(
      (agent) => nodeStatus[agent] === "done" || nodeStatus[agent] === "failed"
    );
    const text = selectedAgents
      .map((agent) => `${LABELS[agent]} ${AGENT_SYMBOL[nodeStatus[agent]] || "●"}`)
      .join(", ");
    steps.push({
      id: "agents",
      // A failed agent is settled too, but the group is not complete while
      // any selected agent is still active. Individual failures stay beside
      // the relevant agent instead of turning the whole row red.
      status: allSettled ? "done" : "active",
      text: `Selected agents: ${text}`,
    });
  }

  if (nodeStatus.synthesizer === "active") {
    steps.push({
      id: "synthesizer",
      status: "active",
      text: "Synthesizer is preparing the final report…",
    });
  } else if (nodeStatus.synthesizer === "done") {
    steps.push({
      id: "synthesizer",
      status: "done",
      text: "The final report is completed.",
    });
  }

  if (nodeStatus.validator === "active") {
    steps.push({
      id: "validator",
      status: "active",
      text: "Validator is checking claims and citations…",
    });
  } else if (nodeStatus.validator === "done") {
    const validationStatus = String(run?.validationStatus || "").toLowerCase();
    let status = "done";
    let text = "Validation is completed.";
    if (validationStatus === "passed") {
      text = "Validation completed — no issues found.";
    } else if (validationStatus === "warnings") {
      text = "Validation completed with warnings.";
    } else if (validationStatus.includes("fail")) {
      status = "failed";
      text = "Validation failed; the final report is still available.";
    }
    steps.push({ id: "validator", status, text });
  }

  return steps;
}
