import test from "node:test";
import assert from "node:assert/strict";

import {
  aggregateRunTokenUsage,
  emptyRun,
  hydrateRun,
  reduce,
} from "./useAnalysisStream.js";
import { buildProgressSteps } from "./progress.js";
import { withoutFactCheck } from "./markdown.js";


test("older snapshots gain the validator state without losing saved nodes", () => {
  const run = hydrateRun({ nodeStatus: { planner: "done", synthesizer: "done" } });

  assert.equal(run.nodeStatus.planner, "done");
  assert.equal(run.nodeStatus.synthesizer, "done");
  assert.equal(run.nodeStatus.validator, "idle");
});


test("node metrics are stored and repeated runs accumulate", () => {
  const first = reduce(emptyRun(), {
    event: "node",
    node: "planner",
    update: {
      info_request: { question: "Which market?" },
      component_metrics: {
        planner: {
          run_count: 1,
          duration_seconds: 0.25,
          token_usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
          },
        },
      },
    },
  });
  const run = reduce(first, {
    event: "node",
    node: "planner",
    update: {
      plan: {},
      component_metrics: {
        planner: {
          run_count: 1,
          duration_seconds: 0.75,
          token_usage: {
            input_tokens: 20,
            output_tokens: 3,
            total_tokens: 23,
          },
        },
      },
    },
  });

  assert.deepEqual(run.componentMetrics.planner, {
    run_count: 2,
    duration_seconds: 1,
    token_usage: {
      input_tokens: 30,
      output_tokens: 5,
      total_tokens: 35,
    },
  });
});


test("run duration accumulates clarification and completion segments", () => {
  const paused = reduce(emptyRun(), {
    event: "clarification",
    question: "Which market?",
    thread_id: "thread-1",
    duration_seconds: 1.25,
  });
  const done = reduce(paused, {
    event: "done",
    duration_seconds: 0.75,
  });

  assert.equal(paused.runDurationSeconds, 1.25);
  assert.equal(done.runDurationSeconds, 2);
  assert.equal(done.status, "done");
});


test("run token totals sum component totals once", () => {
  const totals = aggregateRunTokenUsage({
    planner: {
      token_usage: {
        input_tokens: 10,
        output_tokens: 2,
        total_tokens: 12,
      },
    },
    economics: {
      token_usage: {
        input_tokens: 20,
        output_tokens: 5,
        total_tokens: 25,
      },
      agents: {
        analyst: {
          token_usage: {
            input_tokens: 20,
            output_tokens: 5,
            total_tokens: 25,
          },
        },
      },
    },
  });

  assert.deepEqual(totals, {
    input_tokens: 30,
    output_tokens: 7,
    total_tokens: 37,
  });
});


test("synthesizer completion activates validator", () => {
  const run = reduce(emptyRun(), {
    event: "node",
    node: "synthesizer",
    update: { final_markdown: "# Draft", final_summary: "Draft" },
  });

  assert.equal(run.nodeStatus.synthesizer, "done");
  assert.equal(run.nodeStatus.validator, "active");
  assert.equal(run.finalMarkdown, "# Draft");
});


test("validator completion preserves the synthesis and stores details separately", () => {
  const draft = reduce(emptyRun(), {
    event: "node",
    node: "synthesizer",
    update: { final_markdown: "# Draft" },
  });
  const run = reduce(draft, {
    event: "node",
    node: "validator",
    update: {
      final_markdown: "# Draft\n\n## Fact Check",
      validation_status: "warnings",
      validation_issues: [{ category: "citation_gap" }],
      validation_markdown: "## Fact Check",
      validation_payload: { status: "warnings" },
    },
  });

  assert.equal(run.nodeStatus.validator, "done");
  assert.equal(run.finalMarkdown, "# Draft");
  assert.equal(run.validationStatus, "warnings");
  assert.deepEqual(run.validationIssues, [{ category: "citation_gap" }]);
  assert.equal(run.validationMarkdown, "## Fact Check");
  assert.deepEqual(run.validationPayload, { status: "warnings" });
});


test("legacy appended Fact Check is hidden from report Markdown", () => {
  const legacy = "# Report\n\n## Sources\n\n1. Source\n\n## Fact Check\n\nWarnings";

  assert.equal(withoutFactCheck(legacy), "# Report\n\n## Sources\n\n1. Source");
  assert.equal(hydrateRun({ finalMarkdown: legacy }).finalMarkdown, "# Report\n\n## Sources\n\n1. Source");
});


test("progress starts with the planner and explains clarification", () => {
  const running = reduce(emptyRun(), { event: "start" });
  assert.deepEqual(buildProgressSteps(running), [
    {
      id: "planner",
      status: "active",
      text: "Planner is preparing the analysis plan…",
    },
  ]);

  const clarification = reduce(running, {
    event: "clarification",
    question: "Which market?",
    thread_id: "thread-1",
  });
  assert.equal(
    buildProgressSteps(clarification)[0].text,
    "Planner needs more information to complete the analysis plan."
  );
});


test("selected agents stay on one ordered progress line and update in place", () => {
  const run = emptyRun();
  run.status = "running";
  run.plan = { legal: "Review regulation", economics: "Review costs", political: "Review policy" };
  run.nodeStatus = {
    ...run.nodeStatus,
    planner: "done",
    political: "done",
    economics: "active",
    legal: "failed",
  };

  const steps = buildProgressSteps(run);
  assert.equal(steps.length, 2);
  assert.deepEqual(steps[1], {
    id: "agents",
    status: "active",
    text: "Selected agents: Political ✓, Economic ●, Legal ×",
  });

  run.nodeStatus.economics = "failed";
  assert.equal(buildProgressSteps(run)[1].status, "done");
});


test("progress moves from synthesis completion to validation", () => {
  const run = emptyRun();
  run.status = "running";
  run.nodeStatus.synthesizer = "done";
  run.nodeStatus.validator = "active";

  assert.deepEqual(buildProgressSteps(run), [
    { id: "synthesizer", status: "done", text: "The final report is completed." },
    { id: "validator", status: "active", text: "Validator is checking claims and citations…" },
  ]);

  run.nodeStatus.validator = "done";
  run.validationStatus = "warnings";
  assert.equal(buildProgressSteps(run)[1].text, "Validation completed with warnings.");
});
