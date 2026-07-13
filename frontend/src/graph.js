// Fixed PESTEL topology: planner -> agents fan-out -> synthesizer.
// The backend graph is stable, so we draw it client-side and light nodes up
// from SSE events rather than discovering the shape at runtime.

export const AGENTS = [
  "political",
  "economics",
  "social",
  "technological",
  "environmental",
  "legal",
];

// Order the run moves through, used to lay out and to animate the active path.
export const FLOW = ["planner", ...AGENTS, "synthesizer"];

export const LABELS = {
  planner: "Planner",
  synthesizer: "Synthesizer",
  political: "Political",
  economics: "Economic",
  social: "Social",
  technological: "Technological",
  environmental: "Environmental",
  legal: "Legal",
};
